package handlers

import (
	"net/http"
	"strconv"

	"facego/internal/person"

	"github.com/gin-gonic/gin"
)

type PersonsHandler struct {
	service *person.Service
}

func NewPersonsHandler(svc *person.Service) *PersonsHandler {
	return &PersonsHandler{service: svc}
}

func (h *PersonsHandler) List(c *gin.Context) {
	persons, err := h.service.List()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if persons == nil {
		persons = []person.Person{}
	}
	c.JSON(http.StatusOK, gin.H{"persons": persons, "total": len(persons)})
}

func (h *PersonsHandler) GetByID(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid id"})
		return
	}
	p, err := h.service.GetByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if p == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "person not found"})
		return
	}
	c.JSON(http.StatusOK, p)
}

func (h *PersonsHandler) Delete(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid id"})
		return
	}
	if err := h.service.Delete(id); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"success": true})
}
