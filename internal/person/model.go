package person

import "time"

type Person struct {
	ID        int64     `json:"id"`
	UUID      string    `json:"uuid"`
	Name      string    `json:"name"`
	PhotoPath string    `json:"photo_path,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Embedding struct {
	ID        int64
	PersonID  int64
	Data      []byte
	Model     string
	CreatedAt time.Time
}

type EmbeddingEntry struct {
	PersonID  int64
	Name      string
	UUID      string
	PhotoPath string
	Data      []float32
}
