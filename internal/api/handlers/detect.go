package handlers

import (
	"net/http"

	"facego/internal/face"

	"github.com/gin-gonic/gin"
)

type HeadPose struct {
	Yaw   float64 `json:"yaw"`   // negative = turned left, positive = right
	Pitch float64 `json:"pitch"` // negative = looking up, positive = down
}

// estimatePose derives yaw and pitch from 5 SCRFD landmarks.
// Landmark order: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
func estimatePose(lm face.Landmark) HeadPose {
	eyeCX := (float64(lm.Points[0][0]) + float64(lm.Points[1][0])) / 2
	eyeCY := (float64(lm.Points[0][1]) + float64(lm.Points[1][1])) / 2
	eyeW := float64(lm.Points[1][0]) - float64(lm.Points[0][0])
	mouthY := (float64(lm.Points[3][1]) + float64(lm.Points[4][1])) / 2
	noseX := float64(lm.Points[2][0])
	noseY := float64(lm.Points[2][1])
	faceH := mouthY - eyeCY

	yaw := 0.0
	pitch := 0.0
	if eyeW > 0 {
		yaw = (noseX - eyeCX) / eyeW
	}
	if faceH > 0 {
		// neutral ~0.45; up = smaller; down = larger
		pitch = (noseY-eyeCY)/faceH - 0.45
	}
	return HeadPose{Yaw: yaw, Pitch: pitch}
}

type DetectHandler struct {
	detector face.Detector
}

func NewDetectHandler(detector face.Detector) *DetectHandler {
	return &DetectHandler{detector: detector}
}

func (h *DetectHandler) Detect(c *gin.Context) {
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "image required"})
		return
	}

	f, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot open image"})
		return
	}
	defer f.Close()

	data := make([]byte, file.Size)
	if _, err := f.Read(data); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot read image"})
		return
	}

	result, err := h.detector.Detect(data)
	if err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error()})
		return
	}

	resp := gin.H{
		"detected":   result.Count > 0,
		"faces":      result.Count,
		"confidence": result.Confidence,
	}

	if result.Count == 1 && len(result.Landmarks) > 0 {
		lm := result.Landmarks[0]
		pose := estimatePose(lm)
		resp["landmarks"] = lm.Points
		resp["pose"] = pose
	}

	c.JSON(http.StatusOK, resp)
}
