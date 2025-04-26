import numpy as np

def normalize_embedding(embedding):
    """ Normalize vector for cosine similarity """
    return embedding / np.linalg.norm(embedding)

def prepare_embedding_batch(embeddings):
    """ Stack and normalize a batch of embeddings """
    return np.stack([normalize_embedding(e) for e in embeddings])
