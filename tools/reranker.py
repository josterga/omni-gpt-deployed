from typing import List, Dict
from models.data_models import RetrievedContext

class Reranker:
    def __init__(self, user_weights: Dict[str, float]=None, source_weights: Dict[str, float]=None):
        self.user_weights = user_weights or {}
        self.source_weights = source_weights or {"docs": 1, "discourse": 1, "slack": 1}

    def rerank(self, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        # Handle empty or None contexts
        if not contexts:
            return []
        
        # Validate and fix scores
        for ctx in contexts:
            # Ensure score is a valid number
            if not hasattr(ctx, 'score') or ctx.score is None:
                ctx.score = 0.0
            elif not isinstance(ctx.score, (int, float)):
                try:
                    ctx.score = float(ctx.score)
                except (ValueError, TypeError):
                    ctx.score = 0.0
            
            # Apply weights
            uw = self.user_weights.get(ctx.metadata.get('user',''), 1.0)
            sw = self.source_weights.get(ctx.source, 1.0)
            ctx.score = ctx.score * uw * sw
        
        # Sort by score descending
        return sorted(contexts, key=lambda x: x.score, reverse=True)