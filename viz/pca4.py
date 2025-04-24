import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import faiss
import logging
from pathlib import Path
from config.config_loader import load_config
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class FAISSPCAVisualizer:
    def __init__(self, db_path, metadata_path, refresh_interval=5, time_duration=1.0):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.refresh_interval = refresh_interval
        self.index = None
        self.metadata = []
        self.time_duration = time_duration  # The duration for which the 4th dimension is normalized

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
        self.current_time_point = 0.0  # Start with no visibility (0%) at the beginning

    def load_db(self):
        try:
            self.index = faiss.read_index(str(self.db_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f"Error loading FAISS DB or metadata: {e}")
            return False

    def compute_faiss_pca(self, vectors):
        logging.info("Fitting FAISS PCA...")
        d = vectors.shape[1]
        pca = faiss.PCAMatrix(d, 4)  # We now perform 4D PCA
        pca.train(vectors)
        return pca.apply_py(vectors)

    def run(self):
        angle = 0  # for rotating view

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

            try:
                projected = self.compute_faiss_pca(vectors)
            except Exception as e:
                logging.error(f"PCA projection failed: {e}")
                time.sleep(self.refresh_interval)
                continue

            # Normalize PCA to [0, 1] for all 4 dimensions
            min_vals = projected.min(axis=0)
            max_vals = projected.max(axis=0)
            norm_proj = (projected - min_vals) / (max_vals - min_vals + 1e-8)

            # Extract the first 3 dimensions for the scatter plot
            x, y, z = norm_proj[:, 0], norm_proj[:, 1], norm_proj[:, 2]
            # The fourth dimension (time)
            time_values = norm_proj[:, 3]

            # Normalize the time dimension based on the configured duration
            self.current_time_point = (self.current_time_point + self.refresh_interval / self.time_duration) % 1.0
            time_threshold = self.current_time_point

            # Find the points that correspond to the current time threshold
            time_mask = time_values <= time_threshold  # Reversed: lower time values appear first
            visible_points = norm_proj[time_mask, :]

            # If there are visible points, show them on the plot
            if len(visible_points) > 0:
                # Extract the first 3 dimensions for the visible points
                x_vis, y_vis, z_vis = visible_points[:, 0], visible_points[:, 1], visible_points[:, 2]

                if self.scatter_obj is None:
                    self.scatter_obj = self.ax.scatter(x_vis, y_vis, z_vis, c='white', s=10)
                else:
                    self.scatter_obj._offsets3d = (x_vis, y_vis, z_vis)

            # Rotate the view smoothly from 0 to 360 degrees
            angle = (angle + 1) % 360  # Ensure it loops back to 0 after reaching 360
            self.ax.view_init(elev=30, azim=angle)

            plt.draw()
            plt.pause(0.01)
            time.sleep(self.refresh_interval)

    def cleanup(self):
        logging.info("Shutting down gracefully...")
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)  # Close the plot window
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]

    faiss_db_path = project_root / config["database"]["faiss_index"]
    metadata_path = project_root / config["database"].get("metadata_path", str(faiss_db_path) + ".meta.pkl")

    visualizer = FAISSPCAVisualizer(
        db_path=faiss_db_path,
        metadata_path=metadata_path,
        refresh_interval=0.1,
        time_duration=1.0  # Duration for normalized 4th dimension (time)
    )

    # Handle graceful shutdown with signal
    def signal_handler(sig, frame):
        visualizer.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C interruption

    visualizer.run()
