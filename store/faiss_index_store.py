import faiss
import numpy as np

class FaissIndexStore:
    def __init__(self, dim: int = 1536):  # text-embedding-3-small returns 1536 dimensions
        self.index = faiss.IndexFlatIP(dim)

    def index_query(self, user_query) -> None:
        # user_query.embedding should be a numpy array
        emb = user_query.embedding
        if emb is None:
            return
        
        # Ensure the embedding has the right shape and type
        if isinstance(emb, list):
            emb = np.array(emb, dtype='float32')
        
        vec = emb.reshape(1, -1).astype('float32')
        
        # Check if dimensions match
        if vec.shape[1] != self.index.d:
            print(f"Warning: Embedding dimension {vec.shape[1]} doesn't match index dimension {self.index.d}")
            return
            
        faiss.normalize_L2(vec)
        self.index.add(vec)

    def search(self, emb: np.ndarray, top_k: int=5):
        vec = emb.reshape(1,-1).astype('float32')
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, top_k)
        return D, I