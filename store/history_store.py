import json
from typing import List
from datetime import datetime
from models.data_models import RAGResponse, UserQuery
import logging
from store.session_utils import get_current_session_id

class HistoryStore:
    def __init__(self, path: str):
        self.logger = logging.getLogger("omni_gpt.history_store")
        self.path = path

    def load_sessions(self) -> List[RAGResponse]:
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            self.logger.info("Loaded sessions", extra={"event_type": "HISTORY_LOAD", "details": {"path": self.path, "count": len(data) if isinstance(data, list) else 0, "session_id": get_current_session_id()}})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning("Failed to load sessions", extra={"event_type": "HISTORY_LOAD_FAIL", "details": {"path": self.path, "error": str(e), "session_id": get_current_session_id()}})
            return []

        sessions: List[RAGResponse] = []

        # Determine format: legacy sessions vs flat turns
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'session_id' in data[0]:
            # Legacy structure: list of sessions with 'turns'
            for sess in data:
                for turn in sess.get('turns', []):
                    # Extract text and timestamp
                    user_q_raw = turn.get('user_query')
                    if isinstance(user_q_raw, str):
                        text = user_q_raw
                        ts = turn.get('timestamp_utc')
                    else:
                        text = user_q_raw.get('text', '') if user_q_raw else ''
                        ts = turn.get('timestamp_utc')
                    try:
                        timestamp = datetime.fromisoformat(ts)
                    except Exception:
                        timestamp = datetime.utcnow()
                    user_q = UserQuery(text=text, timestamp=timestamp)

                    answer = turn.get('assistant_response') or turn.get('answer', '')
                    resp = RAGResponse(
                        user_query=user_q,
                        answer=answer,
                        contexts=[],
                        used_cache=False,
                        reranking_info={}
                    )
                    sessions.append(resp)
        elif isinstance(data, list):
            # Flat list of turn records
            for record in data:
                uq = record.get('user_query', {})
                text = uq.get('text', '') if isinstance(uq, dict) else ''
                ts_str = uq.get('timestamp')
                try:
                    timestamp = datetime.fromisoformat(ts_str)
                except Exception:
                    timestamp = datetime.utcnow()
                user_q = UserQuery(text=text, timestamp=timestamp)

                answer = record.get('answer', '')
                used_cache = record.get('used_cache', False)
                rerank = record.get('reranking_info', {})
                resp = RAGResponse(
                    user_query=user_q,
                    answer=answer,
                    contexts=[],
                    used_cache=used_cache,
                    reranking_info=rerank
                )
                sessions.append(resp)

        return sessions

    def save_turn(self, response: RAGResponse) -> None:
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        record = {
            'user_query': {
                'text': response.user_query.text,
                'timestamp': response.user_query.timestamp.isoformat()
            },
            'answer': response.answer,
            'used_cache': response.used_cache,
            'reranking_info': response.reranking_info
        }
        data.append(record)
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved turn", extra={"event_type": "HISTORY_SAVE", "details": {"path": self.path, "session_id": get_current_session_id()}})