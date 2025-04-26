import faiss
import numpy as np
import os
import pickle
import logging

class FAISSDatabase:
    def __init__(self, dim, db_path, metadata_path=None):
        self.db_path = db_path
        self.metadata_path = metadata_path or db_path + ".meta.pkl"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = {}  # Changed to dict keyed by label

        if os.path.exists(db_path):
            self.load()

    def update_metadata(self, label: str, field: str, value):
        if label in self.metadata:
            self.metadata[label][field] = value
            self.save()
        else:
            logging.warning(f"Label {label} not found in metadata.")

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)

        for entry in metadata:
            label = entry["label"]
            self.metadata[label] = entry
        self.save()

    def search(self, query: np.ndarray, k=5):
        distances, indices = self.index.search(query, k)
        results = [list(self.metadata.values())[i] for i in indices[0]]
        return results, distances

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.db_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
