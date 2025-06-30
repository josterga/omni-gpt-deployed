import json
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI
import tiktoken
import re
import logging
from store.session_utils import get_current_session_id

class DocumentSearch:
    def __init__(self, json_path: str, openai_client: OpenAI):
        self.logger = logging.getLogger("omni_gpt.doc_search")
        self.openai = openai_client
        self.metadata = []
        self.embeddings = None
        self.index = None
        try:
            self._load_json(json_path)
            self.logger.info("Loaded document JSON", extra={"event_type": "DOC_JSON_LOAD", "details": {"json_path": json_path}, "session_id": get_current_session_id()})
        except FileNotFoundError:
            self.logger.error("Document JSON file not found", extra={"event_type": "DOC_JSON_NOT_FOUND", "details": {"json_path": json_path}, "session_id": get_current_session_id()})

    def _load_json(self, path: str):
        with open(path) as f:
            docs = json.load(f)
        texts = []
        for d in docs:
            text = d.get('content', '')
            texts.append(text)
            self.metadata.append({
                'id': d.get('id'),
                'title': d.get('title') or "Documentation",
                'content': text,
                'url': d.get('file_name', ''),
                'category': d.get('category', ''),
                'token_count': d.get('token_count', 0),
                'header': d.get('header', ''),
                'source_type': 'docs'
            })
        self.embeddings = self._embed(texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def _embed(self, texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
        try:
            max_batch_tokens = 200_000
            enc = tiktoken.encoding_for_model(model)
            batches, current, tok_count = [], [], 0
            for txt in texts:
                n = len(enc.encode(txt))
                if current and tok_count + n > max_batch_tokens:
                    batches.append(current)
                    current, tok_count = [], 0
                current.append(txt)
                tok_count += n
            if current:
                batches.append(current)
            vectors: list[list[float]] = []
            for batch in batches:
                resp = self.openai.embeddings.create(model=model, input=batch)
                vectors.extend([d.embedding for d in resp.data])
            return np.asarray(vectors, dtype="float32")
        except Exception as e:
            self.logger.error("Embedding failed", extra={"event_type": "DOC_EMBED_ERROR", "details": {"error": str(e)}, "session_id": get_current_session_id()})
            return np.array([])

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.logger.warning("Document search index not initialized", extra={"event_type": "DOC_INDEX_NOT_INIT", "session_id": get_current_session_id()})
            return []
        try:
            q_emb = self._embed([query])
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                meta = self.metadata[idx]
                title = meta.get('title') or "Documentation"
                header = meta.get('header', '')
                combined_text = f"{title} – {header}\n\n{meta.get('content', '')}"
                url = meta.get('url', '')
                if url:
                    clean_path = url.replace('.md', '')
                    clean_path = re.sub(r'(?<=/)[0-9]{2}-', '', clean_path)
                    clean_path = re.sub(r'^s/', '', clean_path)
                    full_url = f"https://docs.omni.co/{clean_path}"
                else:
                    full_url = ""
                results.append({'score': float(score), 'metadata': {**meta, 'title': title}, 'text': combined_text, 'url': full_url})
            self.logger.info("Document search", extra={"event_type": "DOC_SEARCH", "details": {"query": query, "result_count": len(results)}, "session_id": get_current_session_id()})
            return results
        except Exception as e:
            self.logger.error("Document search failed", extra={"event_type": "DOC_SEARCH_ERROR", "details": {"query": query, "error": str(e)}, "session_id": get_current_session_id()})
            return []
