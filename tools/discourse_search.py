import json
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI

class DiscourseSearch:
    def __init__(self, json_path: str, openai_client: OpenAI):
        self.openai = openai_client
        self.metadata = []
        self.embeddings = None
        self.index = None
        try:
            self._load_json(json_path)
        except FileNotFoundError:
            print(f"Warning: Discourse JSON file '{json_path}' not found. Discourse search disabled.")

    def _load_json(self, path: str):
        with open(path) as f:
            docs = json.load(f)
        texts = []
        for d in docs:
            text = d.get('content', '')
            texts.append(text)
            self.metadata.append({
                'id': d.get('source_id'), 
                'title': d.get('title'),
                'content': d.get('content', ''),
                'url': d.get('url', ''),  # Store discourse URL
                'source_type': 'discourse'
            })
        self.embeddings = self._embed(texts)
        dim = self.embeddings.shape[1]  # Use actual embedding dimension
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
        embs = [e.embedding for e in resp.data]
        return np.array(embs, dtype='float32')

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            return []
        q_emb = self._embed([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metadata[idx]
            # Create combined text like the original backup
            combined_text = f"{meta.get('title', '')} {meta.get('content', '')}"
            
            results.append({
                'score': float(score), 
                'metadata': meta,
                'text': combined_text,
                'url': meta.get('url', '')  # Use discourse URL directly
            })
        return results