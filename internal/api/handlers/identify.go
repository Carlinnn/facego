package handlers

import (
	"net/http"

	"facego/internal/face"
	"facego/internal/person"

	"github.com/gin-gonic/gin"
)

type IdentifyHandler struct {
	detector   face.Detector
	recognizer face.Recognizer
	service    *person.Service
	threshold  float64
	maxSize    int64
}

func NewIdentifyHandler(detector face.Detector, recognizer face.Recognizer, svc *person.Service, threshold float64, maxSize int64) *IdentifyHandler {
	return &IdentifyHandler{
		detector:   detector,
		recognizer: recognizer,
		service:    svc,
		threshold:  threshold,
		maxSize:    maxSize,
	}
}

func (h *IdentifyHandler) Identify(c *gin.Context) {
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "image required"})
		return
	}
	if file.Size > h.maxSize {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "image too large"})
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

	detected, err := h.detector.Detect(data)
	if err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error()})
		return
	}
	if detected.Count == 0 {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": "Nenhum rosto foi detectado na imagem."})
		return
	}

	embedding, err := h.recognizer.Embed(data, detected)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot generate embedding"})
		return
	}

	entries := h.service.GetEmbeddings()
	match, similarity := face.FindBestMatch(embedding, entries, h.threshold)

	if match == nil {
		c.JSON(http.StatusOK, gin.H{
			"identified": false,
			"person":     nil,
			"message":    "Rosto não identificado.",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"identified": true,
		"person": gin.H{
			"id":    match.PersonID,
			"uuid":  match.UUID,
			"name":  match.Name,
			"photo": "/faces/" + match.UUID + "/face.jpg",
		},
		"similarity": similarity,
	})
}
