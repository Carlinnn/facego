package database

import (
	"database/sql"
	"fmt"
)

func Migrate(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS persons (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			uuid       TEXT NOT NULL UNIQUE,
			name       TEXT NOT NULL,
			photo_path TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS face_embeddings (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			person_id  INTEGER NOT NULL,
			embedding  BLOB NOT NULL,
			model      TEXT NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_id);
	`)
	if err != nil {
		return fmt.Errorf("migrate: %w", err)
	}
	return nil
}
