from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import List, Dict
import numpy as np
import faiss
import re
import tiktoken
import os
import logging
from store.session_utils import get_current_session_id

class SlackSearch:
    def __init__(self, token: str, openai_client):
        self.logger = logging.getLogger("omni_gpt.slack_search")
        if not token or token == "None":
            print("Warning: No Slack token provided. Slack search will be disabled.")
            self.client = None
        else:
            self.client = WebClient(token=token)
        self.encoder = tiktoken.encoding_for_model("text-embedding-3-small")
        self.max_tokens = 250
        self.openai = openai_client

    def intelligent_query_transform(self, user_query: str) -> str:
        transform_prompt = f"""
        Convert this natural language question into an optimized Slack keyword search query.

        Available Slack search operators:
        - from:@username (search messages from specific user)
        - in:#channel (search in specific channel)
        - after:YYYY-MM-DD (messages after date)
        - before:YYYY-MM-DD (messages before date)
        - during:YYYY-MM-DD (messages on specific date)
        - has:link, has:file (messages with attachments)

        User question: "{user_query}"

        Transform this into an effective Slack keyword search query. If the question mentions:
        - A person's name → use from:@username
        - A channel name → use in:#channel
        - Time references → use appropriate date operators
        - File/link mentions → use has: operators

        Return only the optimized search query, no explanation.
        """
        try:
            from openai import OpenAI
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                return user_query
            openai = OpenAI(api_key=openai_key)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at converting natural language to Slack keyword search syntax."},
                    {"role": "user", "content": transform_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            transformed_query = response.choices[0].message.content.strip()
            transformed_query = transformed_query.replace('"', '').replace("'", "")
            return transformed_query
        except Exception as e:
            print(f"Query transformation failed, using original: {e}")
            return user_query

    def enhanced_salt_query(self, user_query: str) -> str:
        salt_terms = "-in:customer-sla-breach -in:customer-triage -in:marketing -in:sales -in:support-overflow -in:omnis -in:tofu -in:customer-membership-alerts -in:optys-qual-haul -in:sigma-compete -in:closed-lost-notifications -in:vector-alerts -in:notifications-alerts -cypress -github -sentry -squadcast -syften"

        if any(word in user_query.lower() for word in ["who", "when", "where", "in channel", "from", "by", "shared", "posted", "yesterday", "last week", "files"]):
            try:
                transformed_query = self.intelligent_query_transform(user_query)
                if len(transformed_query.split()) > 1:
                    return f"{transformed_query} {salt_terms}"
            except Exception as e:
                print(f"Structured extraction failed: {e}")

        try:
            transformed_query = self.intelligent_query_transform(user_query)
            return f"{transformed_query} {salt_terms}"
        except Exception as e:
            print(f"Query transformation failed: {e}")

        salt_terms = "-cypress -github -customer-sla-breach -customer-triage -sentry -squadcast -syften"
        return f"{user_query} {salt_terms}"

    def clean_docs_query(self, user_query: str) -> str:
        slack_operators = ['from:', 'in:', 'after:', 'before:', 'during:', 'has:']
        cleaned_query = user_query
        for operator in slack_operators:
            pattern = rf'\b{re.escape(operator)}\S+\b'
            cleaned_query = re.sub(pattern, '', cleaned_query)
        cleaned_query = ' '.join(cleaned_query.split())
        return cleaned_query if cleaned_query.strip() else user_query

    def search_messages(self, query: str, count: int = 50) -> List[Dict]:
        contexts = []
        if self.client is None:
            self.logger.warning("Slack search disabled - no valid token", extra={"event_type": "SLACK_DISABLED", "session_id": get_current_session_id()})
            return []
        salted_query = self.enhanced_salt_query(query)
        try:
            response = self.client.search_messages(
                query=salted_query,
                highlight=True,
                sort="score",
                sort_dir="desc",
                count=count
            )
            if response and response.get('ok'):
                messages = response.get('messages', {}).get('matches', [])
                if messages:
                    contexts = self.extract_thread_contexts(messages)
                    contexts = self.rank_contexts_by_relevance(query, contexts)
                    self.logger.info("Slack search success", extra={"event_type": "SLACK_SEARCH", "details": {"query": query, "result_count": len(contexts)}, "session_id": get_current_session_id()})
                    return contexts
                self.logger.info("Slack search no results", extra={"event_type": "SLACK_SEARCH", "details": {"query": query, "result_count": 0}, "session_id": get_current_session_id()})
                return contexts
            self.logger.warning("Slack search failed", extra={"event_type": "SLACK_SEARCH_FAIL", "details": {"query": query}, "session_id": get_current_session_id()})
            return []
        except SlackApiError as e:
            self.logger.error("Slack API error", extra={"event_type": "SLACK_API_ERROR", "details": {"query": query, "error": str(e)}, "session_id": get_current_session_id()})
            return []
        except Exception as e:
            self.logger.error("Unexpected error in Slack search", extra={"event_type": "SLACK_SEARCH_ERROR", "details": {"query": query, "error": str(e)}, "session_id": get_current_session_id()})
            return []

    def get_thread_context(self, channel_id: str, thread_ts: str) -> List[Dict]:
        if self.client is None:
            return []
        try:
            response = self.client.conversations_replies(channel=channel_id, ts=thread_ts, limit=20)
            return response.get('messages', [])
        except SlackApiError:
            return []

    def extract_thread_contexts(self, messages: List[Dict]) -> List[Dict]:
        thread_contexts = {}
        standalone_messages = []

        for message in messages:
            thread_ts = message.get('thread_ts')
            if thread_ts:
                if thread_ts not in thread_contexts:
                    full_thread = self.get_thread_context(message.get('channel', {}).get('id'), thread_ts)
                    chunks = self.chunk_thread_by_tokens(full_thread)
                    for i, chunk in enumerate(chunks):
                        thread_contexts[f"{thread_ts}_{i}"] = {
                            'text': chunk,
                            'user': message.get('user', ''),
                            'channel': message.get('channel', {}).get('name', ''),
                            'timestamp': thread_ts,
                            'thread_context': full_thread,
                            'permalink': message.get('permalink', ''),
                            'combined_text': chunk,
                            'source_type': 'slack'
                        }
            else:
                standalone_messages.append(message)

        standalone_contexts = []
        for msg in standalone_messages:
            base_text = self.combine_message_and_thread(msg, [])
            token_len = len(self.encoder.encode(base_text))
            if token_len > self.max_tokens:
                truncated = self.encoder.decode(self.encoder.encode(base_text)[:self.max_tokens])
            else:
                truncated = base_text
            standalone_contexts.append({
                'text': truncated,
                'user': msg.get('user', ''),
                'channel': msg.get('channel', {}).get('name', ''),
                'timestamp': msg.get('ts', ''),
                'thread_context': [],
                'permalink': msg.get('permalink', ''),
                'combined_text': truncated,
                'source_type': 'slack'
            })

        return list(thread_contexts.values()) + standalone_contexts

    def combine_message_and_thread(self, message: Dict, thread_context: List[Dict]) -> str:
        combined_text = f"Channel: {message.get('channel', {}).get('name', '')}\n"
        combined_text += f"User: {message.get('user', '')}\n"
        combined_text += f"Message: {message.get('text', '')}\n"
        if thread_context:
            combined_text += "\nThread Context:\n"
            for reply in thread_context:
                combined_text += f"- {reply.get('user', '')}: {reply.get('text', '')}\n"
        return combined_text.strip()

    def chunk_thread_by_tokens(self, thread: List[Dict]) -> List[str]:
        chunks, current, current_len = [], [], 0
        for msg in thread:
            line = f"- {msg.get('user', '')}: {msg.get('text', '')}"
            toks = self.encoder.encode(line)
            if current_len + len(toks) > self.max_tokens:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += len(toks)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def rank_contexts_by_relevance(self, user_query: str, contexts: List[Dict]) -> List[Dict]:
        if not contexts:
            return contexts

        try:
            query_emb = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=[user_query]
            ).data[0].embedding

            texts = [ctx['combined_text'] for ctx in contexts]
            ctx_embs = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            ).data

            vectors = np.array([d.embedding for d in ctx_embs], dtype=np.float32)
            query_vec = np.array(query_emb, dtype=np.float32).reshape(1, -1)

            faiss.normalize_L2(vectors)
            faiss.normalize_L2(query_vec)
            scores = np.dot(vectors, query_vec.T).squeeze()

            ranked = sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)
            return [ctx for score, ctx in ranked]
        except Exception as e:
            print(f"Ranking failed: {e}")
            return contexts