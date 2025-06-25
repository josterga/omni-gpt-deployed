from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

@dataclass
class UserQuery:
    text: str
    timestamp: datetime
    source: str = "all"
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_text(cls, text: str):
        return cls(text=text, timestamp=datetime.utcnow())

@dataclass
class RetrievedContext:
    text: str
    score: float
    source: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class RAGResponse:
    user_query: UserQuery
    answer: str
    contexts: List[RetrievedContext]
    used_cache: bool = False
    reranking_info: Dict = field(default_factory=dict)
    sources: List[Dict] = field(default_factory=list)