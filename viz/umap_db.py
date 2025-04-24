import time
import pickle
import faiss
import umap
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class UMAPProjectionService:
    def __init__(self, input_index_path, input_meta_path, output_index_path, output_meta_path, dim=3, refresh_interval=10):
        self.input_index_path = Path(input_index_path)
        self.input_meta_path = Path(input_meta_path)
        self.output_index_path = Path(output_index_path)
        self.output_meta_path = Path(output_meta_path)
        self.dim = dim
        self.refresh_interval = refresh_interval

        self.prev_count = -1

        logging.info("Starting UMAPProjectionService...")
        logging.info(f"Input FAISS index: {self.input_index_path}")
        logging.info(f"Input metadata: {self.input_meta_path}")
        logging.info(f"Output FAISS index: {self.output_index_path}")
        logging.info(f"Output metadata: {self.output_meta_path}")

    def run(self):
        while True:
            try:
                # Load input index and metadata
                input_index = faiss.read_index(str(self.input_index_path))
                with open(self.input_meta_path, "rb") as f:
                    input_metadata = pickle.load(f)

                current_count = input_index.ntotal
                if current_count == self.prev_count:
                    logging.info("No new vectors. Waiting...")
                    time.sleep(self.refresh_interval)
                    continue

                logging.info(f"New data detected. Total vectors: {current_count}")
                vectors = input_index.reconstruct_n(0, current_count)

                logging.info("Running UMAP projection...")
                reducer = umap.UMAP(n_components=self.dim)
                embedding = reducer.fit_transform(vectors).astype('float32')

                # Create new FAISS index
                output_index = faiss.IndexFlatL2(self.dim)
                output_index.add(embedding)

                # Add timestamps to metadata
                timestamp = datetime.now().isoformat()
                output_metadata = [(meta[0], timestamp) for meta in input_metadata]

                # Save new index and metadata
                faiss.write_index(output_index, str(self.output_index_path))
                with open(self.output_meta_path, "wb") as f:
                    pickle.dump(output_metadata, f)

                logging.info("UMAP projection and FAISS update completed.")

                self.prev_count = current_count
                time.sleep(self.refresh_interval)

            except Exception as e:
                logging.error(f"Error in UMAPProjectionService: {e}")
                time.sleep(self.refresh_interval)


if __name__ == "__main__":
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]

    service = UMAPProjectionService(
        input_index_path=project_root / config["database"]["faiss_index"],
        input_meta_path=project_root / config["database"].get("metadata_path", str(project_root / config["database"]["faiss_index"]) + ".meta.pkl"),
        output_index_path=project_root / "database" / "faiss_umap.index",
        output_meta_path=project_root / "database" / "faiss_umap.meta.pkl",
        dim=config["umap"].get("dim", 3),
        refresh_interval=config["umap"].get("refresh_interval", 1)
    )

    service.run()
