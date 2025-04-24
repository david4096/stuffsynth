import faiss
import numpy as np
import os
import pickle

class FAISSDatabase:
    def __init__(self, dim, db_path, metadata_path=None):
        self.db_path = db_path
        self.metadata_path = metadata_path or db_path + ".meta.pkl"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        if os.path.exists(db_path):
            self.load()
            
    def update_metadata(self, label: str, field: str, value):
        for meta in self.metadata:
            if meta.get("label") == label:
                meta[field] = value
                self.save()
                return
        logging.warning(f"Label {label} not found in metadata.")



    def add(self, vectors: np.ndarray, metadata: list):
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)
        print(metadata)
        self.save()

    def search(self, query: np.ndarray, k=5):
        distances, indices = self.index.search(query, k)
        results = [self.metadata[i] for i in indices[0]]
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
