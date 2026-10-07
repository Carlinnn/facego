package handlers

import (
	"net/http"
	"strings"

	"facego/internal/face"
	"facego/internal/person"

	"github.com/gin-gonic/gin"
)

type RegisterHandler struct {
	detector   face.Detector
	recognizer face.Recognizer
	service    *person.Service
	maxSize    int64
}

func NewRegisterHandler(detector face.Detector, recognizer face.Recognizer, svc *person.Service, maxSize int64) *RegisterHandler {
	return &RegisterHandler{
		detector:   detector,
		recognizer: recognizer,
		service:    svc,
		maxSize:    maxSize,
	}
}

func (h *RegisterHandler) Register(c *gin.Context) {
	name := strings.TrimSpace(c.PostForm("name"))
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name required"})
		return
	}

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
	if detected.Count > 1 {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": "Mais de um rosto foi detectado. Utilize uma imagem contendo apenas uma pessoa."})
		return
	}

	embedding, err := h.recognizer.Embed(data, detected)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot generate embedding"})
		return
	}

	p, err := h.service.Register(name, data, embedding, h.recognizer.ModelName())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"person": gin.H{
			"id":   p.ID,
			"uuid": p.UUID,
			"name": p.Name,
		},
	})
}
