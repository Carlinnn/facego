package api

import (
	"net/http"

	"facego/internal/api/handlers"
	"facego/internal/config"
	"facego/internal/face"
	"facego/internal/person"

	"github.com/gin-gonic/gin"
)

func NewRouter(cfg *config.Config, personSvc *person.Service, detector face.Detector, recognizer face.Recognizer) *gin.Engine {
	r := gin.Default()

	r.Static("/faces", cfg.FacesDir)
	r.StaticFS("/web", http.Dir("./web"))
	r.GET("/", func(c *gin.Context) { c.Redirect(http.StatusFound, "/web/index.html") })
	r.GET("/register", func(c *gin.Context) { c.Redirect(http.StatusFound, "/web/register.html") })
	r.GET("/identify", func(c *gin.Context) { c.Redirect(http.StatusFound, "/web/identify.html") })
	r.GET("/persons", func(c *gin.Context) { c.Redirect(http.StatusFound, "/web/persons.html") })

	v1 := r.Group("/api/v1")
	{
		v1.GET("/health", handlers.Health)

		ph := handlers.NewPersonsHandler(personSvc)
		v1.GET("/persons", ph.List)
		v1.GET("/persons/:id", ph.GetByID)
		v1.DELETE("/persons/:id", ph.Delete)

		dh := handlers.NewDetectHandler(detector)
		v1.POST("/detect", dh.Detect)

		rh := handlers.NewRegisterHandler(detector, recognizer, personSvc, cfg.MaxImageSize)
		v1.POST("/persons", rh.Register)

		ih := handlers.NewIdentifyHandler(detector, recognizer, personSvc, cfg.Threshold, cfg.MaxImageSize)
		v1.POST("/identify", ih.Identify)
	}

	return r
}
