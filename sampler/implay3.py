import time
import pickle
import numpy as np
import os
import logging
import threading
from pathlib import Path
import sounddevice as sd
import faiss
from config.config_loader import load_config
import librosa
from collections import deque
import sys
from sklearn.decomposition import PCA
from pydub import AudioSegment

# Setup logging
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)

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
        self.playing_queue = deque(maxlen=8)
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

    def extract_audio_features(self, audio_file):
        y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=False)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=0)

        rms = librosa.feature.rms(y=y)[0]
        volume = librosa.amplitude_to_db(rms, ref=np.max)
        left_channel = y[0, :]
        right_channel = y[1, :]
        stereo_separation = np.mean(np.abs(left_channel - right_channel))

        return volume, stereo_separation

    def train_pca(self, vectors, n_components=3):
        pca = PCA(n_components=n_components)
        pca.fit(vectors)
        self.pca_matrix = pca
        logging.info("🧠 Trained PCA.")
        return pca

    def apply_pca(self, vectors):
        if self.pca_matrix is None:
            raise ValueError("PCA not trained.")
        return self.pca_matrix.transform(vectors)

    def get_audio_by_pca(self, pca_vals):
        index = int(pca_vals[0] * len(self.audio_files))
        index = max(0, min(index, len(self.audio_files) - 1))
        return self.audio_files[index]

    def apply_stereo_and_volume(self, audio_data, pca_vals):
        volume_factor = pca_vals[1]
        stereo_factor = pca_vals[2]

        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        elif audio_data.ndim == 2 and audio_data.shape[0] == 1:
            pass
        elif audio_data.ndim == 2 and audio_data.shape[0] == 2:
            pass
        else:
            raise ValueError(f"Unexpected audio format: {audio_data.shape}")

        volume_factor = np.clip(volume_factor, 0, 1)
        audio_data *= volume_factor

        if audio_data.shape[0] == 2:
            left_channel = audio_data[0, :]
            right_channel = audio_data[1, :]
            left_channel *= (1 - stereo_factor)
            right_channel *= (1 + stereo_factor)
            left_channel = np.clip(left_channel, -1, 1)
            right_channel = np.clip(right_channel, -1, 1)
            audio_data = np.vstack([left_channel, right_channel])

        return np.ascontiguousarray(audio_data)

    def load_and_check_audio(self, audio_file, sample_rate):
        try:
            y, sr = librosa.load(audio_file, sr=sample_rate, mono=False, offset=0.0, duration=1.0)
            if y.ndim == 1:
                y = np.expand_dims(y, axis=0)
            elif y.ndim == 2:
                if y.shape[0] == 1:
                    pass
                elif y.shape[0] == 2:
                    pass
                else:
                    raise ValueError(f"Unsupported audio format in file {audio_file}: Expected mono or stereo.")
            elif y.ndim == 3:
                if y.shape[0] == 1 and y.shape[1] == 1:
                    y = np.squeeze(y, axis=(0, 1))
                    y = np.expand_dims(y, axis=0)
                else:
                    raise ValueError(f"Unexpected shape for audio data: {y.shape}")
            else:
                raise ValueError(f"Unsupported audio format in file {audio_file}: Expected mono or stereo.")

            y = np.ascontiguousarray(y)
            return y, sr
        except Exception as e:
            logging.error(f"❌ Error checking audio file {audio_file}: {e}")
            return None, None

    def play_loop(self, audio_file, pca_vals):
        try:
            audio_data, sr = self.load_and_check_audio(audio_file, self.sample_rate)
            if audio_data is None:
                return

            logging.info(f"▶️ Playing: {audio_file}")

            audio_data = self.apply_stereo_and_volume(audio_data, pca_vals)

            # Ensure audio data is contiguous and has correct shape
            if audio_data.ndim == 1:
                audio_data = np.expand_dims(audio_data, axis=0)
            elif audio_data.ndim == 2 and audio_data.shape[0] == 2:
                pass
            else:
                raise ValueError(f"Unexpected audio data shape: {audio_data.shape}")

            # Ensure contiguity and correct shape for sounddevice playback
            audio_data = np.ascontiguousarray(audio_data)

            with sd.OutputStream(channels=audio_data.shape[0], samplerate=sr, blocksize=self.block_size) as stream:
                while not self.stop_all_signal.is_set() and self.playing_threads.get(audio_file):
                    # Transpose to (samples, channels)
                    stream.write(audio_data.T)
        except Exception as e:
            logging.error(f"❌ Error playing {audio_file}: {e}")

    def update_playing_queue(self, new_file, pca_vals):
        if new_file in self.playing_queue:
            logging.info(f"🔁 {new_file} already playing.")
            return

        if len(self.playing_queue) >= self.playing_queue.maxlen:
            old_file = self.playing_queue.popleft()
            self.playing_threads[old_file] = False
            logging.info(f"🛑 Stopped old sound: {old_file}")

        self.playing_queue.append(new_file)
        self.playing_threads[new_file] = True
        logging.info(f"🟢 Starting new sound: {new_file}")

        t = threading.Thread(target=self.play_loop, args=(new_file, pca_vals), daemon=True)
        t.start()
        logging.info(f"🧵 Thread launched for: {new_file} | Active queue: {list(self.playing_queue)}")

    def run(self):
        logging.info("🚀 Audio playback service started.")
        last_reload_time = 0
        last_log_time = 0
        while True:
            self.load_db()

            current_vector_count = self.index.ntotal
            if current_vector_count <= 0:
                logging.info("⏳ No vectors yet...")
                time.sleep(self.refresh_interval)
                continue

            if current_vector_count == self.last_vector_count:
                time.sleep(self.refresh_interval)
                continue

            new_vectors = self.index.reconstruct_n(0, current_vector_count)
            logging.info(f"📈 Detected new vector(s): {current_vector_count - self.last_vector_count} new.")
            self.last_vector_count = current_vector_count

            if self.pca_matrix is None or self.pca_matrix.components_.shape[0] != new_vectors.shape[1]:
                self.train_pca(new_vectors)

            try:
                proj = self.apply_pca(new_vectors)
                norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
                latest_vals = norm[-1]
                logging.info(f"🎧 Latest PCA values: Volume={latest_vals[1]}, Stereo Separation={latest_vals[2]}")
                selected_file = self.get_audio_by_pca(latest_vals)
                self.update_playing_queue(selected_file, latest_vals)
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
        refresh_interval=0.1
    )

    player.run()
