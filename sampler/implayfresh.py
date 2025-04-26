import time
import pickle
import numpy as np
import os
import logging
import sounddevice as sd
import threading
from pathlib import Path
import faiss
from config.config_loader import load_config
import librosa
from collections import deque
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)

# Force flush output
handler.flush = sys.stdout.flush

logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)


class FAISSPCAVisualizerAndAudioPlayer:
    def __init__(self, db_path, metadata_path, audio_dir, refresh_interval=0.1):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.refresh_interval = refresh_interval
        self.index = None
        self.metadata = []
        self.pca_matrix = None
        self.audio_files = []
        self.sample_rate = 44100
        self.block_size = 1024
        self.playing_queue = deque(maxlen=2)
        self.playing_threads = {}
        self.last_vector_count = 0
        self.stop_all_signal = threading.Event()

        self.load_audio_files()

    def load_audio_files(self):
        self.audio_files.clear()
        for file in os.listdir(self.audio_dir):
            if file.endswith(".wav"):
                self.audio_files.append(os.path.join(self.audio_dir, file))
        logging.info(f"🎧 Loaded {len(self.audio_files)} audio files.")

    def load_db(self):
        try:
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            self.index = faiss.read_index(str(self.db_path))
            return True
        except Exception as e:
            logging.error(f"❌ Error loading FAISS DB or metadata: {e}")
            return False

    def train_pca(self, vectors, n_components=1):
        d = vectors.shape[1]
        pca = faiss.PCAMatrix(d, n_components)
        pca.train(vectors)
        self.pca_matrix = pca
        logging.info("🧠 Trained PCA.")
        return pca

    def apply_pca(self, vectors):
        if self.pca_matrix is None:
            raise ValueError("PCA not trained.")
        return self.pca_matrix.apply(vectors)

    def get_audio_by_pca(self, pca_val):
        index = int(pca_val * len(self.audio_files))
        index = max(0, min(index, len(self.audio_files) - 1))
        return self.audio_files[index]

    def play_loop(self, audio_file):
        volume = 1
        try:
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)
            logging.info(f"▶️ Playing: {audio_file}")
            with sd.OutputStream(channels=1, samplerate=self.sample_rate, blocksize=self.block_size) as stream:
                while not self.stop_all_signal.is_set() and self.playing_threads.get(audio_file):
                    stream.write(audio_data * volume)
        except Exception as e:
            logging.error(f"❌ Error playing {audio_file}: {e}")

    def update_playing_queue(self, new_file):
        if new_file in self.playing_queue:
            print(f"🔁 {new_file} already playing.")
            return

        if len(self.playing_queue) >= self.playing_queue.maxlen:
            old_file = self.playing_queue.popleft()
            self.playing_threads[old_file] = False
            print(f"🛑 Stopped old sound: {old_file}")

        self.playing_queue.append(new_file)
        self.playing_threads[new_file] = True
        print(f"🟢 Starting new sound: {new_file}")
        
        t = threading.Thread(target=self.play_loop, args=(new_file,), daemon=True)
        t.start()
        print(f"🧵 Thread launched for: {new_file} | Active queue: {list(self.playing_queue)}")

    def run(self):
        logging.info("🚀 Audio playback service started.")
        last_played_file = None

        while True:
            self.load_db()

            current_vector_count = self.index.ntotal
            if current_vector_count <= 0:
                logging.info("⏳ No vectors yet...")
                time.sleep(self.refresh_interval)
                continue

            new_vectors = self.index.reconstruct_n(0, current_vector_count)

            if self.pca_matrix is None or self.pca_matrix.d_in != new_vectors.shape[1]:
                self.train_pca(new_vectors)

            try:
                proj = self.apply_pca(new_vectors)
                norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
                latest_val = norm[-1][0]
                selected_file = self.get_audio_by_pca(latest_val)

                if selected_file != last_played_file:
                    self.update_playing_queue(selected_file)
                    last_played_file = selected_file
                else:
                    # If already playing, ensure it stays active (could trigger reset logic if needed)
                    if selected_file not in self.playing_queue:
                        self.update_playing_queue(selected_file)

            except Exception as e:
                logging.error(f"❌ PCA or audio update failed: {e}")

            time.sleep(self.refresh_interval)



if __name__ == "__main__":
    config = load_config()
    root = Path(__file__).resolve().parents[1]
    faiss_db = root / config["database"]["faiss_index"]
    meta = root / config["database"].get("metadata_path", str(faiss_db) + ".meta.pkl")
    audio_dir = root / config["audio"]["wav_directory"]

    player = FAISSPCAVisualizerAndAudioPlayer(
        db_path=faiss_db,
        metadata_path=meta,
        audio_dir=audio_dir,
        refresh_interval=0.1,
    )
    player.run()
