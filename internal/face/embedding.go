package face

import "facego/internal/person"

// FindBestMatch returns best matching entry above threshold, or nil.
func FindBestMatch(embedding []float32, entries []person.EmbeddingEntry, threshold float64) (*person.EmbeddingEntry, float64) {
	var best *person.EmbeddingEntry
	bestSim := -1.0

	for i := range entries {
		sim := CosineSimilarity(embedding, entries[i].Data)
		if sim > bestSim {
			bestSim = sim
			best = &entries[i]
		}
	}

	if best == nil || bestSim < threshold {
		return nil, bestSim
	}
	return best, bestSim
}
