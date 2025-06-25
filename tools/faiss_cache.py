import faiss
import numpy as np

class FaissIndexStore:
    def __init__(self, dim: int = 768):
        self.index = faiss.IndexFlatIP(dim)

    def index_query(self, user_query):
        emb = user_query.embedding
        vec = emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(vec)
        self.index.add(vec)

    def search(self, emb: np.ndarray, top_k: int=5):
        vec = emb.reshape(1,-1).astype('float32')
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, top_k)
        return D, I