import sounddevice as sd
import numpy as np
import queue
import threading
import time
import faiss
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class PCA_Synthesizer:
    def __init__(self, faiss_db_path, metadata_path, sample_rate=44100, block_size=1024, tween_steps=10, refresh_interval=0.1):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vector_queue = queue.Queue()
        self.last_pca_vector = np.zeros(3)  # Shape (3,)
        self.target_pca_vector = np.zeros(3)  # Shape (3,)
        self.vector_lock = threading.Lock()
        self.tween_steps = tween_steps
        self.current_tween_step = 0
        self.tween_vector = np.zeros(3)  # Shape (3,)
        self.refresh_interval = refresh_interval

        self.load_faiss_db(faiss_db_path, metadata_path)
        
        # Start the audio stream
        self.stream = sd.Stream(callback=self.stream_callback, channels=1, samplerate=self.sample_rate, blocksize=self.block_size)

    def load_faiss_db(self, faiss_db_path, metadata_path):
        try:
            # Load the FAISS index and metadata
            self.index = faiss.read_index(str(faiss_db_path))
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logging.info("FAISS database and metadata loaded.")
        except Exception as e:
            logging.error(f"Error loading FAISS DB or metadata: {e}")
            raise

    def compute_faiss_pca(self, vectors):
        logging.info("Fitting FAISS PCA...")
        d = vectors.shape[1]
        pca = faiss.PCAMatrix(d, 3)  # Reduce to 3 dimensions
        pca.train(vectors)
        return pca.apply_py(vectors)

    def get_latest_vector(self):
        try:
            # Get the latest vector from the FAISS database
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            if vectors.shape[0] > 0:
                # Apply PCA transformation to the vectors
                projected_vectors = self.compute_faiss_pca(vectors)
                latest_vector = projected_vectors[-1]  # Get the latest projected vector
                return latest_vector
        except Exception as e:
            logging.error(f"Failed to fetch vectors from FAISS: {e}")
        return np.zeros(3)  # Default if unable to fetch

    def generate_tone(self, pca_values):
        frequency = 440 * (1 + pca_values[0])  # Modulate pitch
        amplitude = 0.5 * (1 + pca_values[1])  # Modulate amplitude
        time_stretch = 1 + pca_values[2]       # Modulate time stretch

        # Generate a sine wave based on the PCA values
        t = np.linspace(0, self.block_size / self.sample_rate, self.block_size)
        tone = amplitude * np.sin(2 * np.pi * frequency * t)

        # Ensure tone length matches block size
        if len(tone) < self.block_size:
            tone = np.tile(tone, int(np.ceil(self.block_size / len(tone))))[:self.block_size]
        
        return tone

    def tween_vectors(self):
        # Linear interpolation between last_pca_vector and target_pca_vector
        if self.current_tween_step < self.tween_steps:
            # Ensure tween_vector remains shape (3,)
            self.tween_vector = self.last_pca_vector + (self.target_pca_vector - self.last_pca_vector) * (self.current_tween_step / self.tween_steps)
            self.current_tween_step += 1
        else:
            # Once the tween is done, update the last_pca_vector and set a new target
            self.last_pca_vector = self.target_pca_vector
            self.current_tween_step = 0

    def stream_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        # Acquire the lock to ensure thread-safe access to vector data
        with self.vector_lock:
            # Fetch the latest PCA vector from the FAISS database
            self.target_pca_vector = self.get_latest_vector()

        # Tween between the last and target PCA vectors
        self.tween_vectors()

        # Generate tone based on tweened PCA values
        audio_data = self.generate_tone(self.tween_vector)

        # Check for NaN or infinite values in audio data
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            print("Warning: Audio data contains NaN or Inf values. Skipping this block.")
            return

        # Ensure audio_data is of correct type and shape
        audio_data = np.float32(audio_data)

        # Ensure the output matches the required frame size
        audio_data = np.tile(audio_data, int(np.ceil(frames / len(audio_data))))[:frames]

        # Output the audio data
        outdata[:] = np.asarray(audio_data, dtype=np.float32).reshape(-1, 1)

    def run(self):
        with self.stream:
            print("Streaming audio. Press Ctrl+C to stop.")
            while True:
                time.sleep(self.refresh_interval)

if __name__ == "__main__":
    # Define paths to your FAISS database and metadata
    faiss_db_path = Path('path_to_your_faiss_index_file')
    metadata_path = Path('path_to_your_metadata_file')

    synthesizer = PCA_Synthesizer(faiss_db_path=faiss_db_path, metadata_path=metadata_path)

    # Start the audio synthesis
    synthesizer.run()
