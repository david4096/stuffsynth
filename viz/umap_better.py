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

class UMAPRenderer:
    def __init__(self, umap_index_path, umap_metadata_path, refresh_interval=5):
        self.umap_index_path = umap_index_path
        self.umap_metadata_path = umap_metadata_path
        self.refresh_interval = refresh_interval
        self.index = None
        self.metadata = []

        plt.ion()
        self.fig = plt.figure(facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        self.ax.xaxis.pane.set_color((0, 0, 0, 1))
        self.ax.yaxis.pane.set_color((0, 0, 0, 1))
        self.ax.zaxis.pane.set_color((0, 0, 0, 1))
        self.ax.tick_params(colors='white')

    def load_db(self):
        try:
            self.index = faiss.read_index(str(self.umap_index_path))
            with open(self.umap_metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f"Error loading UMAP FAISS DB or metadata: {e}")
            return False

    def run(self):
        while True:
            if not self.load_db():
                logging.warning("Waiting for UMAP FAISS database...")
                time.sleep(self.refresh_interval)
                continue

            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            if vectors.shape[1] != 3:
                logging.warning("UMAP vectors are not 3D. Skipping render.")
                time.sleep(self.refresh_interval)
                continue

            self.ax.clear()
            self.ax.set_facecolor('black')

            xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
            self.ax.scatter(xs[:-1], ys[:-1], zs[:-1], c='white', s=10)
            self.ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=30)  # most recent

            self.ax.set_title("UMAP Projection", color='white')
            plt.draw()
            plt.pause(0.01)
            time.sleep(self.refresh_interval)

if __name__ == "__main__":
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]

    umap_index_path = project_root / config["umap"]["faiss_index"]
    umap_metadata_path = project_root / config["umap"]["metadata_path"]

    renderer = UMAPRenderer(
        umap_index_path=umap_index_path,
        umap_metadata_path=umap_metadata_path,
        refresh_interval=5
    )
    renderer.run()
