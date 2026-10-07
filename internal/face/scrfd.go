package face

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	scrfdInputSize = 640
	scrfdMean      = 127.5
	scrfdStd       = 128.0
)

type faceCandidate struct {
	score float32
	box   BoundingBox
	lm    Landmark
}

// SCRFDDetector detects faces using the SCRFD det_10g ONNX model.
type SCRFDDetector struct {
	modelPath   string
	scoreThresh float32
	nmsThresh   float32
}

func NewSCRFDDetector(modelPath string, scoreThresh, nmsThresh float32) *SCRFDDetector {
	return &SCRFDDetector{
		modelPath:   modelPath,
		scoreThresh: scoreThresh,
		nmsThresh:   nmsThresh,
	}
}

// strideDef maps model output names to stride values.
// Order: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
var (
	scrfdOutputNames = []string{"448", "471", "494", "451", "474", "497", "454", "477", "500"}
	scrfdStrides     = []int{8, 16, 32}
	scrfdNumAnchors  = 2
)

func (d *SCRFDDetector) Detect(imageData []byte) (*DetectResult, error) {
	img, err := decodeImage(imageData)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	resized, scale, _, padX, padY := letterboxResize(img, scrfdInputSize, scrfdInputSize)
	input := imageToNCHW(resized, scrfdMean, scrfdStd)

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, scrfdInputSize, scrfdInputSize), input)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Output shapes: score=[N,1], bbox=[N,4], kps=[N,10] where N=fh*fw*numAnchors
	// Counts: stride8=12800, stride16=3200, stride32=800
	counts := make([]int64, 3)
	for i, s := range scrfdStrides {
		fh := scrfdInputSize / s
		counts[i] = int64(fh * (scrfdInputSize / s) * scrfdNumAnchors)
	}

	makeScore := func(n int64) (*ort.Tensor[float32], error) {
		return ort.NewEmptyTensor[float32](ort.NewShape(n, 1))
	}
	makeBbox := func(n int64) (*ort.Tensor[float32], error) {
		return ort.NewEmptyTensor[float32](ort.NewShape(n, 4))
	}
	makeKps := func(n int64) (*ort.Tensor[float32], error) {
		return ort.NewEmptyTensor[float32](ort.NewShape(n, 10))
	}

	s0, _ := makeScore(counts[0]); defer s0.Destroy()
	s1, _ := makeScore(counts[1]); defer s1.Destroy()
	s2, _ := makeScore(counts[2]); defer s2.Destroy()
	b0, _ := makeBbox(counts[0]); defer b0.Destroy()
	b1, _ := makeBbox(counts[1]); defer b1.Destroy()
	b2, _ := makeBbox(counts[2]); defer b2.Destroy()
	k0, _ := makeKps(counts[0]); defer k0.Destroy()
	k1, _ := makeKps(counts[1]); defer k1.Destroy()
	k2, _ := makeKps(counts[2]); defer k2.Destroy()

	session, err := ort.NewAdvancedSession(d.modelPath,
		[]string{"input.1"},
		scrfdOutputNames,
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{s0, s1, s2, b0, b1, b2, k0, k1, k2},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run: %w", err)
	}

	scoreTensors := []*ort.Tensor[float32]{s0, s1, s2}
	bboxTensors := []*ort.Tensor[float32]{b0, b1, b2}
	kpsTensors := []*ort.Tensor[float32]{k0, k1, k2}

	var candidates []faceCandidate

	for si, stride := range scrfdStrides {
		fh := scrfdInputSize / stride
		fw := scrfdInputSize / stride
		scores := scoreTensors[si].GetData()
		bboxes := bboxTensors[si].GetData()
		kps := kpsTensors[si].GetData()

		idx := 0
		for y := 0; y < fh; y++ {
			for x := 0; x < fw; x++ {
				for a := 0; a < scrfdNumAnchors; a++ {
					score := scores[idx] // model already applies sigmoid
					if score >= d.scoreThresh {
						cx := (float32(x) + 0.5) * float32(stride)
						cy := (float32(y) + 0.5) * float32(stride)
						b := bboxes[idx*4:]
						x1 := (cx - b[0]*float32(stride) - float32(padX)) / scale
						y1 := (cy - b[1]*float32(stride) - float32(padY)) / scale
						x2 := (cx + b[2]*float32(stride) - float32(padX)) / scale
						y2 := (cy + b[3]*float32(stride) - float32(padY)) / scale
						x1 = clampF(x1, 0, float32(origW))
						y1 = clampF(y1, 0, float32(origH))
						x2 = clampF(x2, 0, float32(origW))
						y2 = clampF(y2, 0, float32(origH))

						var lm Landmark
						k := kps[idx*10:]
						for p := 0; p < 5; p++ {
							px := (cx + k[p*2]*float32(stride) - float32(padX)) / scale
							py := (cy + k[p*2+1]*float32(stride) - float32(padY)) / scale
							lm.Points[p] = [2]float32{px, py}
						}
						candidates = append(candidates, faceCandidate{score: score,
							box: BoundingBox{X1: x1, Y1: y1, X2: x2, Y2: y2}, lm: lm})
					}
					idx++
				}
			}
		}
	}

	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	kept := nmsFilter(candidates, d.nmsThresh)

	result := &DetectResult{Count: len(kept)}
	for _, c := range kept {
		result.Boxes = append(result.Boxes, c.box)
		result.Landmarks = append(result.Landmarks, c.lm)
		if float64(c.score) > result.Confidence {
			result.Confidence = float64(c.score)
		}
	}
	return result, nil
}

func nmsFilter(candidates []faceCandidate, thresh float32) []faceCandidate {
	suppressed := make([]bool, len(candidates))
	var kept []faceCandidate
	for i := range candidates {
		if suppressed[i] {
			continue
		}
		kept = append(kept, candidates[i])
		for j := i + 1; j < len(candidates); j++ {
			if !suppressed[j] && iou(candidates[i].box, candidates[j].box) > thresh {
				suppressed[j] = true
			}
		}
	}
	return kept
}

func iou(a, b BoundingBox) float32 {
	ix1 := max32(a.X1, b.X1)
	iy1 := max32(a.Y1, b.Y1)
	ix2 := min32(a.X2, b.X2)
	iy2 := min32(a.Y2, b.Y2)
	inter := max32(0, ix2-ix1) * max32(0, iy2-iy1)
	areaA := (a.X2 - a.X1) * (a.Y2 - a.Y1)
	areaB := (b.X2 - b.X1) * (b.Y2 - b.Y1)
	union := areaA + areaB - inter
	if union == 0 {
		return 0
	}
	return inter / union
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func decodeImage(data []byte) (image.Image, error) {
	r := bytes.NewReader(data)
	img, _, err := image.Decode(r)
	if err == nil {
		return img, nil
	}
	r.Seek(0, 0)
	img, err = jpeg.Decode(r)
	if err == nil {
		return img, nil
	}
	r.Seek(0, 0)
	return png.Decode(r)
}

func letterboxResize(img image.Image, dstW, dstH int) (image.Image, float32, float32, int, int) {
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()
	scaleX := float32(dstW) / float32(origW)
	scaleY := float32(dstH) / float32(origH)
	scale := scaleX
	if scaleY < scale {
		scale = scaleY
	}
	newW := int(float32(origW) * scale)
	newH := int(float32(origH) * scale)
	padX := (dstW - newW) / 2
	padY := (dstH - newH) / 2

	out := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	gray := color.RGBA{114, 114, 114, 255}
	for y := 0; y < dstH; y++ {
		for x := 0; x < dstW; x++ {
			out.Set(x, y, gray)
		}
	}
	for y := 0; y < newH; y++ {
		srcY := int(float32(y) / scale)
		if srcY >= origH {
			srcY = origH - 1
		}
		for x := 0; x < newW; x++ {
			srcX := int(float32(x) / scale)
			if srcX >= origW {
				srcX = origW - 1
			}
			out.Set(x+padX, y+padY, img.At(srcX, srcY))
		}
	}
	return out, scale, scale, padX, padY
}

func imageToNCHW(img image.Image, mean, std float64) []float32 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	data := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			data[0*h*w+y*w+x] = float32((float64(b>>8) - mean) / std) // B
			data[1*h*w+y*w+x] = float32((float64(g>>8) - mean) / std) // G
			data[2*h*w+y*w+x] = float32((float64(r>>8) - mean) / std) // R
		}
	}
	return data
}
