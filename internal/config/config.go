package config

import (
	"os"
	"strconv"
)

type Config struct {
	Port          string
	DatabasePath  string
	FacesDir      string
	ModelsDir     string
	Threshold     float64
	MaxImageSize  int64
}

func Load() *Config {
	return &Config{
		Port:         getEnv("PORT", "8080"),
		DatabasePath: getEnv("DATABASE_PATH", "./data/facego.db"),
		FacesDir:     getEnv("FACES_DIR", "./data/faces"),
		ModelsDir:    getEnv("MODELS_DIR", "./models"),
		Threshold:    getEnvFloat("THRESHOLD", 0.5),
		MaxImageSize: getEnvInt64("MAX_IMAGE_SIZE", 10*1024*1024), // 10MB
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

func getEnvInt64(key string, fallback int64) int64 {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.ParseInt(v, 10, 64); err == nil {
			return i
		}
	}
	return fallback
}
