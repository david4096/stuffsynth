import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import faiss
import umap
import logging
from pathlib import Path
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class FAISSVisualizer:
    def __init__(self, db_path, metadata_path, refresh_interval=5):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.refresh_interval = refresh_interval
        self.index = None
        self.metadata = []

        logging.info("Initializing FAISS visualizer...")
        logging.info(f"FAISS index path: {self.db_path}")
        logging.info(f"Metadata path: {self.metadata_path}")

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def load_db(self):
        try:
            logging.info("Loading FAISS index...")
            self.index = faiss.read_index(str(self.db_path))
            logging.info("FAISS index loaded successfully.")

            logging.info("Loading metadata...")
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logging.info(f"Metadata loaded successfully. Entries: {len(self.metadata)}")
            return True
        except Exception as e:
            logging.error(f"Error loading FAISS DB or metadata: {e}")
            return False

    def run(self):
        while True:
            if not self.load_db():
                logging.warning("Waiting for FAISS database...")
                time.sleep(self.refresh_interval)
                continue

            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            logging.info(f"Reconstructed {vectors.shape[0]} vectors from FAISS.")

            if vectors.shape[0] < 3:
                logging.warning("Not enough points to visualize. Need at least 3.")
                time.sleep(self.refresh_interval)
                continue

            logging.info("Running UMAP projection...")
            try:
                embedding = umap.UMAP(n_components=3).fit_transform(vectors)
            except Exception as e:
                logging.error(f"UMAP projection failed: {e}")
                time.sleep(self.refresh_interval)
                continue

            self.ax.clear()
            self.ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='blue', s=10)

            for i, label in enumerate(self.metadata):
                try:
                    self.ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], label[0], fontsize=6)
                except Exception as e:
                    logging.warning(f"Could not label point {i}: {e}")

            plt.draw()
            plt.pause(0.01)
            time.sleep(self.refresh_interval)


if __name__ == "__main__":
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]  # adjust if necessary

    faiss_db_path = project_root / config["database"]["faiss_index"]
    metadata_path = project_root / config["database"].get("metadata_path", str(faiss_db_path) + ".meta.pkl")

    visualizer = FAISSVisualizer(
        db_path=faiss_db_path,
        metadata_path=metadata_path,
        refresh_interval=5
    )
    visualizer.run()
