import os
import librosa
import numpy as np
from config.config_loader import load_config

import logging
from pathlib import Path
from database.faiss_db import FAISSDatabase
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Feature extraction method - using MFCC as an example
def extract_mfcc_features(wav_file, n_mfcc=13, max_len=512):
    """Extract MFCC features from a WAV file."""
    try:
        y, sr = librosa.load(wav_file, sr=None)  # Load audio
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Average over time to reduce the dimension
        mfcc = np.mean(mfcc, axis=1)
        # Ensure the length is 512, or pad if needed
        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)), mode='constant')
        elif len(mfcc) > max_len:
            mfcc = mfcc[:max_len]
        return mfcc
    except Exception as e:
        logging.error(f"Error extracting MFCC features from {wav_file}: {e}")
        return None

def process_wav_directory(config, embedding_dim=512):
    """Process all WAV files in a directory, generate embeddings, and store them in a FAISS index."""
    wav_directory = Path(config['audio']['wav_directory'])
    faiss_db_path = config['audio']['faiss_index']
    metadata_path = config['audio']['metadata_path']

    wav_files = [f for f in os.listdir(wav_directory) if f.endswith('.wav')]
    logging.info(f"Found {len(wav_files)} WAV files in {wav_directory}")

    # Initialize the FAISS database
    faiss_db = FAISSDatabase(dim=embedding_dim, db_path=faiss_db_path, metadata_path=metadata_path)

    embeddings = []
    metadata = []

    for wav_file in wav_files:
        wav_path = os.path.join(wav_directory, wav_file)
        logging.info(f"Processing {wav_path}...")

        # Extract features
        features = extract_mfcc_features(wav_path)
        if features is not None:
            embeddings.append(features)
            metadata.append(wav_file)  # Use the file name as metadata

    # Convert to numpy array and add to FAISS index
    embeddings = np.array(embeddings).astype(np.float32)
    faiss_db.add(embeddings, metadata)
    logging.info(f"Added {len(wav_files)} files to FAISS index.")

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Process the directory and generate FAISS embeddings
    process_wav_directory(config, embedding_dim=config['audio']['embedding_dim'])

    logging.info("FAISS index generation complete.")
