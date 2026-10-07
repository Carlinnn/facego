package face

import (
	"fmt"
	"image"
	"image/draw"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

const arcfaceSize = 112

// ArcFace reference landmarks for 112x112 alignment
var arcfaceRef = [5][2]float32{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// ArcFaceRecognizer generates face embeddings using ArcFace ONNX model.
type ArcFaceRecognizer struct {
	modelPath string
}

func NewArcFaceRecognizer(modelPath string) *ArcFaceRecognizer {
	return &ArcFaceRecognizer{modelPath: modelPath}
}

func (r *ArcFaceRecognizer) ModelName() string {
	return "arcface-w600k-r50"
}

func (r *ArcFaceRecognizer) Embed(imageData []byte, detection *DetectResult) ([]float32, error) {
	if len(detection.Landmarks) == 0 {
		return nil, fmt.Errorf("no landmarks for alignment")
	}

	img, err := decodeImage(imageData)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	aligned := alignFace(img, detection.Landmarks[0])
	input := faceToNCHW(aligned)

	inputShape := ort.NewShape(1, 3, arcfaceSize, arcfaceSize)
	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 512)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(r.modelPath,
		[]string{"input.1"},
		[]string{"683"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	embedding := outputTensor.GetData()
	result := make([]float32, len(embedding))
	copy(result, embedding)
	return NormalizeL2(result), nil
}

// alignFace crops and warps face to 112x112 using 5-point similarity transform.
func alignFace(img image.Image, lm Landmark) image.Image {
	// Compute similarity transform from lm.Points -> arcfaceRef
	M := similarityTransform(lm.Points, arcfaceRef)

	out := image.NewRGBA(image.Rect(0, 0, arcfaceSize, arcfaceSize))
	draw.Draw(out, out.Bounds(), image.Black, image.Point{}, draw.Src)

	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	// Inverse transform: for each dst pixel find src pixel
	det := M[0][0]*M[1][1] - M[0][1]*M[1][0]
	if math.Abs(det) < 1e-10 {
		return out
	}
	invM := [2][3]float64{
		{M[1][1] / det, -M[0][1] / det, (M[0][1]*M[1][2] - M[1][1]*M[0][2]) / det},
		{-M[1][0] / det, M[0][0] / det, (M[1][0]*M[0][2] - M[0][0]*M[1][2]) / det},
	}

	for dy := 0; dy < arcfaceSize; dy++ {
		for dx := 0; dx < arcfaceSize; dx++ {
			sx := invM[0][0]*float64(dx) + invM[0][1]*float64(dy) + invM[0][2]
			sy := invM[1][0]*float64(dx) + invM[1][1]*float64(dy) + invM[1][2]
			ix := int(sx + 0.5)
			iy := int(sy + 0.5)
			if ix >= 0 && ix < srcW && iy >= 0 && iy < srcH {
				out.Set(dx, dy, img.At(ix+bounds.Min.X, iy+bounds.Min.Y))
			}
		}
	}
	return out
}

// similarityTransform computes 2x3 affine matrix mapping src 5 points to dst.
// Uses least-squares similarity transform (uniform scale, rotation, translation).
func similarityTransform(src, dst [5][2]float32) [2][3]float64 {
	// Build system using umeyama-style estimation
	n := 5
	var srcX, srcY, dstX, dstY float64
	for i := 0; i < n; i++ {
		srcX += float64(src[i][0])
		srcY += float64(src[i][1])
		dstX += float64(dst[i][0])
		dstY += float64(dst[i][1])
	}
	srcX /= float64(n)
	srcY /= float64(n)
	dstX /= float64(n)
	dstY /= float64(n)

	var a, b, c float64
	for i := 0; i < n; i++ {
		sx := float64(src[i][0]) - srcX
		sy := float64(src[i][1]) - srcY
		dx := float64(dst[i][0]) - dstX
		dy := float64(dst[i][1]) - dstY
		a += sx*dx + sy*dy
		b += sx*dy - sy*dx
		c += sx*sx + sy*sy
	}
	if c == 0 {
		c = 1
	}
	scale := math.Sqrt(a*a+b*b) / c
	cos := a / (c * scale)
	sin := b / (c * scale)

	tx := dstX - scale*(cos*srcX-sin*srcY)
	ty := dstY - scale*(sin*srcX+cos*srcY)

	return [2][3]float64{
		{scale * cos, -scale * sin, tx},
		{scale * sin, scale * cos, ty},
	}
}

// faceToNCHW converts 112x112 face image to ArcFace input (RGB, normalized to [-1,1]).
func faceToNCHW(img image.Image) []float32 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	data := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// normalize to [-1, 1]: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
			rf := float32(float64(r>>8)/127.5 - 1.0)
			gf := float32(float64(g>>8)/127.5 - 1.0)
			bf := float32(float64(b>>8)/127.5 - 1.0)
			// RGB channel order for ArcFace
			data[0*h*w+y*w+x] = rf
			data[1*h*w+y*w+x] = gf
			data[2*h*w+y*w+x] = bf
		}
	}
	return data
}
