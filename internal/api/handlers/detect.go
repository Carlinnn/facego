package handlers

import (
	"net/http"

	"facego/internal/face"

	"github.com/gin-gonic/gin"
)

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

	c.JSON(http.StatusOK, gin.H{
		"detected":   result.Count > 0,
		"faces":      result.Count,
		"confidence": result.Confidence,
	})
}
