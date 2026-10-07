package face

// DetectResult holds detection output.
type DetectResult struct {
	Count      int
	Confidence float64
	Boxes      []BoundingBox
	Landmarks  []Landmark
}

// BoundingBox is a detected face region.
type BoundingBox struct {
	X1, Y1, X2, Y2 float32
}

// Landmark holds 5-point facial landmarks.
type Landmark struct {
	Points [5][2]float32
}

// Detector detects faces in raw image bytes.
type Detector interface {
	Detect(imageData []byte) (*DetectResult, error)
}

// Recognizer generates face embeddings.
type Recognizer interface {
	Embed(imageData []byte, detection *DetectResult) ([]float32, error)
	ModelName() string
}
