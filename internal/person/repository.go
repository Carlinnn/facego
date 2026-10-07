package person

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"time"
)

type Repository struct {
	db *sql.DB
}

func NewRepository(db *sql.DB) *Repository {
	return &Repository{db: db}
}

func (r *Repository) Create(p *Person) error {
	_, err := r.db.Exec(
		`INSERT INTO persons (uuid, name, photo_path) VALUES (?, ?, ?)`,
		p.UUID, p.Name, p.PhotoPath,
	)
	if err != nil {
		return fmt.Errorf("create person: %w", err)
	}
	row := r.db.QueryRow(`SELECT id, created_at, updated_at FROM persons WHERE uuid = ?`, p.UUID)
	return row.Scan(&p.ID, &p.CreatedAt, &p.UpdatedAt)
}

func (r *Repository) List() ([]Person, error) {
	rows, err := r.db.Query(`SELECT id, uuid, name, photo_path, created_at, updated_at FROM persons ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var persons []Person
	for rows.Next() {
		var p Person
		var photo sql.NullString
		if err := rows.Scan(&p.ID, &p.UUID, &p.Name, &photo, &p.CreatedAt, &p.UpdatedAt); err != nil {
			return nil, err
		}
		p.PhotoPath = photo.String
		persons = append(persons, p)
	}
	return persons, rows.Err()
}

func (r *Repository) GetByID(id int64) (*Person, error) {
	var p Person
	var photo sql.NullString
	err := r.db.QueryRow(
		`SELECT id, uuid, name, photo_path, created_at, updated_at FROM persons WHERE id = ?`, id,
	).Scan(&p.ID, &p.UUID, &p.Name, &photo, &p.CreatedAt, &p.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	p.PhotoPath = photo.String
	return &p, nil
}

func (r *Repository) Delete(id int64) error {
	_, err := r.db.Exec(`DELETE FROM persons WHERE id = ?`, id)
	return err
}

func (r *Repository) UpdatePhoto(id int64, path string) error {
	_, err := r.db.Exec(
		`UPDATE persons SET photo_path = ?, updated_at = ? WHERE id = ?`,
		path, time.Now(), id,
	)
	return err
}

func (r *Repository) SaveEmbedding(personID int64, embedding []float32, model string) error {
	blob := float32SliceToBytes(embedding)
	_, err := r.db.Exec(
		`INSERT INTO face_embeddings (person_id, embedding, model) VALUES (?, ?, ?)`,
		personID, blob, model,
	)
	return err
}

func (r *Repository) LoadAllEmbeddings() ([]EmbeddingEntry, error) {
	rows, err := r.db.Query(`
		SELECT p.id, p.name, p.uuid, p.photo_path, e.embedding
		FROM persons p
		JOIN face_embeddings e ON e.person_id = p.id
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []EmbeddingEntry
	for rows.Next() {
		var e EmbeddingEntry
		var blob []byte
		var photo sql.NullString
		if err := rows.Scan(&e.PersonID, &e.Name, &e.UUID, &photo, &blob); err != nil {
			return nil, err
		}
		e.PhotoPath = photo.String
		e.Data = bytesToFloat32Slice(blob)
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

func (r *Repository) Count() (int, error) {
	var count int
	err := r.db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&count)
	return count, err
}

func float32SliceToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func bytesToFloat32Slice(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return f
}
