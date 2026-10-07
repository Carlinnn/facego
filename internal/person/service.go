package person

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

type Service struct {
	repo      *Repository
	facesDir  string
	mu        sync.RWMutex
	embeddings []EmbeddingEntry
}

func NewService(repo *Repository, facesDir string) *Service {
	return &Service{repo: repo, facesDir: facesDir}
}

func (s *Service) Register(name string, imageData []byte, embedding []float32, model string) (*Person, error) {
	id := uuid.New().String()

	p := &Person{UUID: id, Name: name}
	if err := s.repo.Create(p); err != nil {
		return nil, fmt.Errorf("create person: %w", err)
	}

	dir := filepath.Join(s.facesDir, id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create face dir: %w", err)
	}

	photoPath := filepath.Join(dir, "face.jpg")
	if err := os.WriteFile(photoPath, imageData, 0644); err != nil {
		return nil, fmt.Errorf("save photo: %w", err)
	}

	if err := s.repo.UpdatePhoto(p.ID, photoPath); err != nil {
		return nil, fmt.Errorf("update photo path: %w", err)
	}
	p.PhotoPath = photoPath

	if err := s.repo.SaveEmbedding(p.ID, embedding, model); err != nil {
		return nil, fmt.Errorf("save embedding: %w", err)
	}

	s.mu.Lock()
	s.embeddings = append(s.embeddings, EmbeddingEntry{
		PersonID:  p.ID,
		Name:      p.Name,
		UUID:      p.UUID,
		PhotoPath: p.PhotoPath,
		Data:      embedding,
	})
	s.mu.Unlock()

	return p, nil
}

func (s *Service) List() ([]Person, error) {
	return s.repo.List()
}

func (s *Service) GetByID(id int64) (*Person, error) {
	return s.repo.GetByID(id)
}

func (s *Service) Delete(id int64) error {
	p, err := s.repo.GetByID(id)
	if err != nil {
		return err
	}
	if p == nil {
		return fmt.Errorf("person not found")
	}

	if err := s.repo.Delete(id); err != nil {
		return err
	}

	if p.PhotoPath != "" {
		dir := filepath.Dir(p.PhotoPath)
		_ = os.RemoveAll(dir)
	}

	s.mu.Lock()
	filtered := s.embeddings[:0]
	for _, e := range s.embeddings {
		if e.PersonID != id {
			filtered = append(filtered, e)
		}
	}
	s.embeddings = filtered
	s.mu.Unlock()

	return nil
}

func (s *Service) LoadEmbeddings() error {
	entries, err := s.repo.LoadAllEmbeddings()
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.embeddings = entries
	s.mu.Unlock()
	return nil
}

func (s *Service) GetEmbeddings() []EmbeddingEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]EmbeddingEntry, len(s.embeddings))
	copy(result, s.embeddings)
	return result
}

func (s *Service) Count() (int, error) {
	return s.repo.Count()
}
