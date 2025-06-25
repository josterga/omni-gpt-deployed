from typing import List, Dict
from models.data_models import RetrievedContext

class Reranker:
    def __init__(self, user_weights: Dict[str, float]=None, source_weights: Dict[str, float]=None):
        self.user_weights = user_weights or {}
        self.source_weights = source_weights or {"docs": 3.0, "discourse": 2.0, "slack": 1.0}

    def rerank(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        for ctx in contexts:
            uw = self.user_weights.get(ctx.metadata.get('user',''), 1.0)
            sw = self.source_weights.get(ctx.source, 1.0)
            ctx.score = ctx.score * uw * sw
        return sorted(contexts, key=lambda x: x.score, reverse=True)