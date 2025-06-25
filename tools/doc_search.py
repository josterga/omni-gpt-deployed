import json
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI
import tiktoken
import re

class DocumentSearch:
    def __init__(self, json_path: str, openai_client: OpenAI):
        self.openai = openai_client
        self.metadata = []
        self.embeddings = None
        self.index = None
        try:
            self._load_json(json_path)
        except FileNotFoundError:
            print(f"Warning: Document JSON file '{json_path}' not found. Document search disabled.")

    def _load_json(self, path: str):
        with open(path) as f:
            docs = json.load(f)
        texts = []
        for d in docs:
            text = d.get('content', '')
            texts.append(text)
            self.metadata.append({
                'id': d.get('id'), 
                'title': d.get('title'),
                'content': d.get('content', ''),
                'url': d.get('file_name', ''),  # Store file_name as URL
                'category': d.get('category', ''),
                'source_type': 'docs'
            })
        self.embeddings = self._embed(texts)
        dim = self.embeddings.shape[1]  # Use actual embedding dimension
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def _embed(self, texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
        max_batch_tokens = 200_000                  # leave ~100k safety margin
        enc              = tiktoken.encoding_for_model(model)

        # ── split texts into token-safe batches ────────────────────────────────
        batches, current, tok_count = [], [], 0
        for txt in texts:
            n = len(enc.encode(txt))
            # if adding this text would overflow the batch, start a new one
            if current and tok_count + n > max_batch_tokens:
                batches.append(current)
                current, tok_count = [], 0
            current.append(txt)
            tok_count += n
        if current:
            batches.append(current)

        # ── embed each batch and concatenate results ──────────────────────────
        vectors: list[list[float]] = []
        for batch in batches:
            resp = self.openai.embeddings.create(model=model, input=batch)
            vectors.extend([d.embedding for d in resp.data])

        return np.asarray(vectors, dtype="float32")

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
            
            # Create proper URL for docs
            url = meta.get('url', '')
            if url:
                # Remove .md extension
                clean_path = url.replace('.md', '')
                # Remove any /##- after the first directory
                clean_path = re.sub(r'(?<=/)[0-9]{2}-', '', clean_path)
                # Remove leading s/ if present
                clean_path = re.sub(r'^s/', '', clean_path)
                full_url = f"https://docs.omni.co/{clean_path}"
            else:
                full_url = ""
            
            results.append({
                'score': float(score), 
                'metadata': meta,
                'text': combined_text,
                'url': full_url
            })
        return results