import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import faiss
import logging
from pathlib import Path
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class FAISSPCAVisualizer:
    def __init__(self, db_path, metadata_path, refresh_interval=5):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.refresh_interval = refresh_interval
        self.index = None
        self.metadata = []
        self.pca_matrix = None  # Store PCA matrix for transformation

        plt.ion()
        self.fig = plt.figure(figsize=(12, 10), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')

        # Make axes invisible and remove ticks
        self.ax.set_axis_off()
        self.ax.grid(False)

        # Maximize the figure window (works on macOS with TkAgg backend)
        if plt.get_backend() == "TkAgg":
            self.fig.canvas.manager.window.attributes('-zoomed', 1)

        self.scatter_obj = None
        self.latest_point = None

    def load_db(self):
        try:
            self.index = faiss.read_index(str(self.db_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f"Error loading FAISS DB or metadata: {e}")
            return False

    def train_pca(self, vectors):
        logging.info("Fitting FAISS PCA...")
        d = vectors.shape[1]
        pca = faiss.PCAMatrix(d, 3)  # 3 is the target dimension for PCA
        pca.train(vectors)
        self.pca_matrix = pca
        return pca.apply_py(vectors)

    def apply_pca(self, vectors):
        if self.pca_matrix is None:
            raise ValueError("PCA has not been trained.")
        return self.pca_matrix.apply_py(vectors)

    def run(self):
        # Set initial fixed axes limits based on your data range
        ax_limit = 1.0  # Set a reasonable axis limit based on your data range
        self.ax.set_xlim(-ax_limit, ax_limit)
        self.ax.set_ylim(-ax_limit, ax_limit)
        self.ax.set_zlim(-ax_limit, ax_limit)

        # Initial vectors and PCA training
        while True:
            if not self.load_db():
                logging.warning("Waiting for FAISS database...")
                time.sleep(self.refresh_interval)
                continue

            try:
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception as e:
                logging.error(f"Failed to reconstruct vectors: {e}")
                time.sleep(self.refresh_interval)
                continue

            if vectors.shape[0] < 3:
                logging.warning("Not enough vectors for PCA.")
                time.sleep(self.refresh_interval)
                continue

            # Train PCA only once on initial data
            if self.pca_matrix is None:
                try:
                    projected = self.train_pca(vectors)
                except Exception as e:
                    logging.error(f"PCA projection failed: {e}")
                    time.sleep(self.refresh_interval)
                    continue
            else:
                # Apply previously trained PCA to new vectors
                try:
                    projected = self.apply_pca(vectors)
                except Exception as e:
                    logging.error(f"PCA projection failed: {e}")
                    time.sleep(self.refresh_interval)
                    continue

            # Normalize PCA to [0, 1]
            min_vals = projected.min(axis=0)
            max_vals = projected.max(axis=0)
            norm_proj = (projected - min_vals) / (max_vals - min_vals + 1e-8)

            x, y, z = norm_proj[:-1, 0], norm_proj[:-1, 1], norm_proj[:-1, 2]
            latest = norm_proj[-1, :]

            if self.scatter_obj is None:
                self.scatter_obj = self.ax.scatter(x, y, z, c='white', s=10)
                self.latest_point = self.ax.scatter(*latest, c='red', s=20)
            else:
                self.scatter_obj._offsets3d = (x, y, z)
                self.latest_point._offsets3d = ([latest[0]], [latest[1]], [latest[2]])

            plt.draw()
            plt.pause(0.01)
            time.sleep(self.refresh_interval)

if __name__ == "__main__":
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]

    faiss_db_path = config['audio']['faiss_index']
    metadata_path = config['audio']['metadata_path']

    visualizer = FAISSPCAVisualizer(
        db_path=faiss_db_path,
        metadata_path=metadata_path,
        refresh_interval=0.1
    )
    visualizer.run()
