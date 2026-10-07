package face

import "fmt"

// StubDetector returns error until real ONNX models are loaded.
// Replace with SCRFDDetector in Phase 2.
type StubDetector struct{}

func (s *StubDetector) Detect(_ []byte) (*DetectResult, error) {
	return nil, fmt.Errorf("face detection not available: models not loaded")
}

// StubRecognizer returns error until ArcFace model is loaded.
// Replace with ArcFaceRecognizer in Phase 3.
type StubRecognizer struct{}

func (s *StubRecognizer) Embed(_ []byte, _ *DetectResult) ([]float32, error) {
	return nil, fmt.Errorf("face recognition not available: models not loaded")
}

func (s *StubRecognizer) ModelName() string {
	return "arcface-stub"
}
