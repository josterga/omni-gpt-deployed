import json
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI
import logging
from store.session_utils import get_current_session_id

class DiscourseSearch:
    def __init__(self, json_path: str, openai_client: OpenAI):
        self.logger = logging.getLogger("omni_gpt.discourse_search")
        self.openai = openai_client
        self.metadata = []
        self.embeddings = None
        self.index = None
        try:
            self._load_json(json_path)
            self.logger.info("Loaded discourse JSON", extra={"event_type": "DISCOURSE_JSON_LOAD", "details": {"json_path": json_path}, "session_id": get_current_session_id()})
        except FileNotFoundError:
            self.logger.error("Discourse JSON file not found", extra={"event_type": "DISCOURSE_JSON_NOT_FOUND", "details": {"json_path": json_path}, "session_id": get_current_session_id()})

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
        try:
            resp = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
            embs = [e.embedding for e in resp.data]
            return np.array(embs, dtype='float32')
        except Exception as e:
            self.logger.error("Discourse embedding failed", extra={"event_type": "DISCOURSE_EMBED_ERROR", "details": {"error": str(e)}, "session_id": get_current_session_id()})
            return np.array([])

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.logger.warning("Discourse search index not initialized", extra={"event_type": "DISCOURSE_INDEX_NOT_INIT", "session_id": get_current_session_id()})
            return []
        try:
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
            self.logger.info("Discourse search", extra={"event_type": "DISCOURSE_SEARCH", "details": {"query": query, "result_count": len(results)}, "session_id": get_current_session_id()})
            return results
        except Exception as e:
            self.logger.error("Discourse search failed", extra={"event_type": "DISCOURSE_SEARCH_ERROR", "details": {"query": query, "error": str(e)}, "session_id": get_current_session_id()})
            return []