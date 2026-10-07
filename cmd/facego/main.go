package main

import (
	"log"
	"os"
	"path/filepath"

	"facego/internal/api"
	"facego/internal/config"
	"facego/internal/database"
	"facego/internal/face"
	"facego/internal/person"
)

func main() {
	cfg := config.Load()

	if err := os.MkdirAll(filepath.Dir(cfg.DatabasePath), 0755); err != nil {
		log.Fatalf("create data dir: %v", err)
	}
	if err := os.MkdirAll(cfg.FacesDir, 0755); err != nil {
		log.Fatalf("create faces dir: %v", err)
	}

	db, err := database.Open(cfg.DatabasePath)
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer db.Close()

	if err := database.Migrate(db); err != nil {
		log.Fatalf("migrate: %v", err)
	}

	repo := person.NewRepository(db)
	personSvc := person.NewService(repo, cfg.FacesDir)

	if err := personSvc.LoadEmbeddings(); err != nil {
		log.Fatalf("load embeddings: %v", err)
	}

	scrfdPath := filepath.Join(cfg.ModelsDir, "scrfd.onnx")
	arcfacePath := filepath.Join(cfg.ModelsDir, "arcface.onnx")

	var detector face.Detector
	var recognizer face.Recognizer

	if _, err := os.Stat(scrfdPath); err == nil {
		if _, err := os.Stat(arcfacePath); err == nil {
			if err := face.InitORT(""); err != nil {
				log.Printf("WARNING: ONNX Runtime init failed: %v", err)
				log.Printf("Running with stub detector — face recognition disabled")
				detector = &face.StubDetector{}
				recognizer = &face.StubRecognizer{}
			} else {
				log.Printf("Loading SCRFD from %s", scrfdPath)
				detector = face.NewSCRFDDetector(scrfdPath, 0.5, 0.4)
				log.Printf("Loading ArcFace from %s", arcfacePath)
				recognizer = face.NewArcFaceRecognizer(arcfacePath)
				log.Printf("Face recognition enabled")
			}
		} else {
			log.Printf("WARNING: ArcFace model not found at %s", arcfacePath)
			detector = &face.StubDetector{}
			recognizer = &face.StubRecognizer{}
		}
	} else {
		log.Printf("WARNING: SCRFD model not found at %s", scrfdPath)
		log.Printf("To enable face recognition, place models in %s", cfg.ModelsDir)
		detector = &face.StubDetector{}
		recognizer = &face.StubRecognizer{}
	}

	router := api.NewRouter(cfg, personSvc, detector, recognizer)

	log.Printf("FaceGo iniciando na porta %s", cfg.Port)
	if err := router.Run(":" + cfg.Port); err != nil {
		log.Fatalf("server: %v", err)
	}
}
