import streamlit as st
import re
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from typing import List, Dict
import json
from datetime import datetime, timedelta, timezone
import uuid

# ALLOWED_EMAIL_DOMAIN = "omni.co"

# def require_login():
#     user = st.user
#     st.write("🔍 [debug] `st.user` object:", user)  # Debug output

#     if user is None:
#         st.warning("⚠️ [debug] No user detected — user not signed in yet.")
#         st.stop()

#     email = user.get("email", "")
#     st.write("📧 [debug] Detected user email:", email)  # Debug email

#     if not email.endswith(f"@{ALLOWED_EMAIL_DOMAIN}"):
#         st.error("🚫 You are not authorized to use this app.")
#         st.warning(f"⚠️ [debug] Unauthorized domain: {email}")
#         st.stop()

# Where we’ll store everything
CHAT_LOG_PATH = "chat_history.json"
load_dotenv()

def _load_history(path: str) -> list[dict]:
    """Return whole file as list[session] or empty list."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass  # corrupted/on first run → fall through
    return []

def _save_history(history: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def _new_session() -> dict:
    """Start a fresh session block."""
    return {
        "session_id": str(uuid.uuid4()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "turns": []              # will hold individual Q/A pairs
    }


class NaturalLanguageSlackSearch:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def extract_search_attributes(self, query: str) -> Dict:
        """Extract structured attributes from natural language query"""
        
        extraction_prompt = f"""
        Extract search keywords from this natural language query.
        
        Query: "{query}"
        
        Return ONLY a valid JSON object with these fields (use null if not mentioned):
        {{
            "channel": "channel name without # symbol",
            "user": "username without @ symbol", 
            "date_after": "YYYY-MM-DD format if 'after', 'since', 'from date' mentioned",
            "date_before": "YYYY-MM-DD format if 'before', 'until' mentioned",
            "date_on": "YYYY-MM-DD format if specific date mentioned",
            "has_file": true/false if files/attachments mentioned,
            "has_link": true/false if links mentioned,
            "keywords": "main topic/keywords to search for"
        }}
        
        Time references like "last week", "yesterday", "this month" should be converted to actual dates.
        Today's date is {datetime.now().strftime('%Y-%m-%d')}.
        
        Return ONLY the JSON object, no markdown formatting or explanations.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting search parameters. Return ONLY valid JSON without markdown formatting."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_content.startswith('```'):
                response_content = response_content[7:]
            if response_content.startswith('```'):
                response_content = response_content[3:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            # Parse JSON
            attributes = json.loads(response_content)
            return attributes
            
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing failed. Raw response: {response_content[:100]}...")
            return {"keywords": query}
        except Exception as e:
            st.warning(f"Attribute extraction failed: {e}")
            return {"keywords": query}

    def build_slack_query(self, attributes: Dict) -> str:
        """Build Slack search query from extracted attributes"""
        query_parts = []
        
        # Add channel filter
        if attributes.get("channel"):
            query_parts.append(f"in:#{attributes['channel']}")
        
        # Add user filter  
        if attributes.get("user"):
            query_parts.append(f"from:@{attributes['user']}")
        
        # Add date filters
        if attributes.get("date_after"):
            query_parts.append(f"after:{attributes['date_after']}")
        
        if attributes.get("date_before"):
            query_parts.append(f"before:{attributes['date_before']}")
            
        if attributes.get("date_on"):
            query_parts.append(f"during:{attributes['date_on']}")
        
        # Add content filters
        if attributes.get("has_file"):
            query_parts.append("has:file")
            
        if attributes.get("has_link"):
            query_parts.append("has:link")
        
        # Add keywords (main search terms)
        if attributes.get("keywords"):
            query_parts.append(attributes["keywords"])
        
        return " ".join(query_parts)
class DiscourseSearchSystem:
    def __init__(self, openai_client, discourse_json_path: str = None):
        self.openai_client = openai_client
        self.discourse_json_path = discourse_json_path
        self.discourse_embeddings = None
        self.discourse_index = None
        self.discourse_metadata = []
        
        if discourse_json_path and os.path.exists(discourse_json_path):
            self.initialize_discourse_search()
    
    def initialize_discourse_search(self):
        """Initialize discourse embeddings and FAISS index from existing JSON"""
        try:
            with open(self.discourse_json_path, 'r') as f:
                discourse_data = json.load(f)
            
            for doc in discourse_data:
                # Extract pre-computed embeddings
                if 'embedding' in doc:
                    embedding = doc['embedding']
                    if isinstance(embedding, list):
                        self.discourse_embeddings = np.array([embedding], dtype='float32') if self.discourse_embeddings is None else np.vstack([self.discourse_embeddings, np.array([embedding], dtype='float32')])
                
                # Build metadata for discourse posts
                self.discourse_metadata.append({
                    'id': doc.get('source_id', ''),
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'source': 'discourse',
                    'url': doc.get('url', ''),
                    'source_type': 'discourse',
                    'combined_text': f"{doc.get('title', '')} {doc.get('content', '')}"
                })
            
            # Initialize FAISS index
            if self.discourse_embeddings is not None and len(self.discourse_embeddings) > 0:
                dimension = self.discourse_embeddings.shape[1]
                self.discourse_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(self.discourse_embeddings)
                self.discourse_index.add(self.discourse_embeddings)
                
        except Exception as e:
            st.error(f"Error initializing discourse search: {e}")
    
    def search_discourse(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search discourse posts using vector embeddings"""
        if self.discourse_index is None:
            return []
            
        query_embedding = self.get_embeddings([query])
        if len(query_embedding) == 0:
            return []
            
        faiss.normalize_L2(query_embedding)
        scores, indices = self.discourse_index.search(query_embedding, min(top_k, len(self.discourse_metadata)))
        
        discourse_results = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            idx = indices[0][i]
            
            if idx != -1 and idx < len(self.discourse_metadata):
                discourse_result = self.discourse_metadata[idx].copy()
                discourse_result['similarity_score'] = float(score)
                discourse_result['rank'] = i + 1
                discourse_results.append(discourse_result)
    
        return discourse_results
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])


class DocumentSearchSystem:
    def __init__(self, openai_client, docs_json_path: str = None):
        self.openai_client = openai_client
        self.docs_json_path = docs_json_path
        self.docs_embeddings = None
        self.docs_index = None
        self.docs_metadata = []
        
        if docs_json_path and os.path.exists(docs_json_path):
            self.initialize_docs_search()
    
    def initialize_docs_search(self):
        """Initialize document embeddings and FAISS index from existing JSON"""
        try:
            with open(self.docs_json_path, 'r') as f:
                docs_data = json.load(f)
            
            doc_texts = []
            for doc in docs_data:
                # Handle different JSON structures
                if 'embedding' in doc:
                    # If embeddings are already stored
                    embedding = doc['embedding']
                    if isinstance(embedding, list):
                        self.docs_embeddings = np.array([embedding], dtype='float32') if self.docs_embeddings is None else np.vstack([self.docs_embeddings, np.array([embedding], dtype='float32')])
                    
                    text_content = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('description', '')}"
                else:
                    # If no embeddings, create text for embedding generation
                    text_content = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('description', '')}"
                
                doc_texts.append(text_content)
                
                self.docs_metadata.append({
                    'id': doc.get('id', ''),
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'source': 'documentation',
                    'url': doc.get('file_name', ''),
                    'category': doc.get('category', ''),
                    'combined_text': text_content,
                    'source_type': 'docs'
                })
            
            # If embeddings weren't pre-stored, generate them
            if self.docs_embeddings is None:
                self.docs_embeddings = self.get_embeddings(doc_texts)
            
            if len(self.docs_embeddings) > 0:
                dimension = self.docs_embeddings.shape[1]
                self.docs_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(self.docs_embeddings)
                self.docs_index.add(self.docs_embeddings)
                
            # st.success(f"✅ Loaded {len(self.docs_metadata)} documents for search")
            
        except Exception as e:
            st.error(f"Error initializing document search: {e}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search documents using vector embeddings"""
        if self.docs_index is None:
            return []
            
        query_embedding = self.get_embeddings([query])
        if len(query_embedding) == 0:
            return []
            
        faiss.normalize_L2(query_embedding)
        scores, indices = self.docs_index.search(query_embedding, min(top_k, len(self.docs_metadata)))
        
        doc_results = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            idx = indices[0][i]
            
            if idx != -1 and idx < len(self.docs_metadata):  # ✅ Now idx is a scalar
                doc_result = self.docs_metadata[idx].copy()
                doc_result['similarity_score'] = float(score)
                doc_result['rank'] = i + 1
                doc_results.append(doc_result)
    
        return doc_results  

class SlackRAGSystem:
    def __init__(self, slack_token: str, openai_api_key: str):
        self.slack_client = WebClient(token=slack_token)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.nl_search = NaturalLanguageSlackSearch(self.openai_client)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def intelligent_query_transform(self, user_query: str) -> str:
        """Transform natural language query into optimized Slack search syntax"""
        
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
            response = self.openai_client.chat.completions.create(
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
            st.warning(f"Query transformation failed, using original: {e}")
            return user_query

    def enhanced_salt_query(self, user_query: str) -> str:
        """Enhanced query processing with multiple strategies and fallback"""
        
        salt_terms = "-in:customer-sla-breach -in:customer-triage -in:marketing -in:sales -in:support-overflow -in:omnis -in:tofu -in:customer-membership-alerts -in:optys-qual-haul -in:sigma-compete -in:closed-lost-notifications -in:vector-alerts -in:notifications-alerts -cypress -github -sentry -squadcast -syften"
        
        # Strategy 1: Try structured extraction for complex queries
        if any(word in user_query.lower() for word in ["who", "when", "where", "in channel", "from", "by", "shared", "posted", "yesterday", "last week", "files"]):
            try:
                attributes = self.nl_search.extract_search_attributes(user_query)
                structured_query = self.nl_search.build_slack_query(attributes)
                
                # If we got meaningful structure, use it
                if len(structured_query.split()) > 1:
                    return f"{structured_query} {salt_terms}"
            except Exception as e:
                st.warning(f"Structured extraction failed: {e}")
        
        # Strategy 2: Simple intelligent transformation
        try:
            transformed_query = self.intelligent_query_transform(user_query)
            return f"{transformed_query} {salt_terms}"
        except Exception as e:
            st.warning(f"Query transformation failed: {e}")
        
        # Strategy 3: Fallback to original with salt
        salt_terms = "-cypress -github -customer-sla-breach -customer-triage -sentry -squadcast -syften"
        return f"{user_query} {salt_terms}"
    
    def clean_docs_query(self, user_query: str) -> str:
        """Clean query processing for documentation search"""
        # Remove Slack-specific operators that don't make sense for docs
        slack_operators = ['from:', 'in:', 'after:', 'before:', 'during:', 'has:']
        
        # Simple cleaning - remove Slack operators
        cleaned_query = user_query
        for operator in slack_operators:
            # Remove operator and the word following it
            pattern = rf'\b{re.escape(operator)}\S+\b'
            cleaned_query = re.sub(pattern, '', cleaned_query)
        
        # Clean up extra spaces
        cleaned_query = ' '.join(cleaned_query.split())
        
        # If query becomes empty after cleaning, use original
        return cleaned_query if cleaned_query.strip() else user_query
        
    def search_slack_messages(self, query: str) -> Dict:
        """Search Slack messages using the Web API"""
        try:
            response = self.slack_client.search_messages(
                query=query,
                highlight=True,
                sort='timestamp',
                sort_dir='desc',
                count=50
            )
            return response
        except SlackApiError as e:
            st.error(f"Error calling Slack API: {e.response['error']}")
            return None
    
    def get_thread_context(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Retrieve thread context for better RAG performance"""
        try:
            response = self.slack_client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=10
            )
            return response.get('messages', [])
        except SlackApiError:
            return []
    
    def extract_thread_contexts(self, messages: List[Dict]) -> List[Dict]:
        """Process threads as complete units rather than individual messages"""
        thread_contexts = {}
        standalone_messages = []
        
        for message in messages:
            thread_ts = message.get('thread_ts')
            
            if thread_ts:
                # This is part of a thread
                if thread_ts not in thread_contexts:
                    # Get full thread context once
                    full_thread = self.get_thread_context(
                        message.get('channel', {}).get('id'), 
                        thread_ts
                    )
                    
                    # Use your existing combine_message_and_thread method
                    combined_text = self.combine_message_and_thread(message, full_thread)
                    
                    # Create consolidated thread context
                    thread_contexts[thread_ts] = {
                        'text': message.get('text', ''),
                        'user': message.get('user', ''),
                        'channel': message.get('channel', {}).get('name', ''),
                        'timestamp': thread_ts,
                        'thread_context': full_thread,
                        'permalink': message.get('permalink', ''),
                        'combined_text': combined_text,
                        'source_type': 'slack'
                    }
            else:
                # Standalone message - process normally
                standalone_messages.append(message)
        
        # Process standalone messages using existing extract_message_context logic
        standalone_contexts = []
        for msg in standalone_messages:
            context = {
                'text': msg.get('text', ''),
                'user': msg.get('user', ''),
                'channel': msg.get('channel', {}).get('name', ''),
                'timestamp': msg.get('ts', ''),
                'thread_context': [],
                'permalink': msg.get('permalink', ''),
                'combined_text': self.combine_message_and_thread(msg, []),
                'source_type': 'slack'
            }
            standalone_contexts.append(context)
        
        # Combine thread contexts and standalone messages
        return list(thread_contexts.values()) + standalone_contexts

    def combine_message_and_thread(self, message: Dict, thread_context: List[Dict]) -> str:
        """Combine message with thread context using smart chunking strategies"""
        combined_text = f"Channel: {message.get('channel', {}).get('name', '')}\n"
        combined_text += f"User: {message.get('user', '')}\n"
        combined_text += f"Message: {message.get('text', '')}\n"
        
        if thread_context:
            combined_text += "\nThread Context:\n"
            for reply in thread_context[:5]:
                combined_text += f"- {reply.get('user', '')}: {reply.get('text', '')}\n"
        
        return combined_text
    
    def rank_contexts_by_relevance(self, user_query: str, contexts: List[Dict]) -> List[Dict]:
        """Rank contexts by semantic similarity using OpenAI embeddings and FAISS"""
        if not contexts:
            return contexts
            
        query_embedding = self.get_embeddings([user_query])
        context_texts = [ctx['combined_text'] for ctx in contexts]
        context_embeddings = self.get_embeddings(context_texts)
        
        if len(context_embeddings) == 0 or len(query_embedding) == 0:
            return contexts
        
        dimension = context_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(context_embeddings)
        faiss.normalize_L2(query_embedding)
        
        index.add(context_embeddings)
        scores, indices = index.search(query_embedding, len(contexts))
        
        ranked_contexts = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            idx = indices[0][i]
        
            if idx != -1 and idx < len(contexts):  # ✅ Now idx is a scalar
                contexts[idx]['similarity_score'] = float(score)
                ranked_contexts.append(contexts[idx])
    
        return ranked_contexts

class UnifiedRAGSystem(SlackRAGSystem):
    def __init__(self, slack_token: str, openai_api_key: str, docs_json_path: str = None, discourse_json_path: str = None):
        super().__init__(slack_token, openai_api_key)
        self.doc_search = DocumentSearchSystem(self.openai_client, docs_json_path) if docs_json_path else None
        self.discourse_search = DiscourseSearchSystem(self.openai_client, discourse_json_path) if discourse_json_path else None
    
    def process_query_by_source(self, user_query: str, source: str) -> str:
        """Process query differently based on search source"""
        if source == "docs":
            # For docs: clean, simple query without Slack-specific terms
            return self.clean_docs_query(user_query)
        elif source == "slack":
            # For Slack: use existing salted query
            return self.enhanced_salt_query(user_query)
        else:  # "both"
            # For unified search: return clean query (will be processed separately)
            return user_query
    
    def search_unified(self, query: str, source: str = "all", top_k: int = 10) -> Dict:
        """Unified search with discourse as third source"""
        results = {
            'slack_results': [],
            'doc_results': [],
            'discourse_results': [],
            'combined_results': []
        }
        
        # Slack search
        if source in ["slack", "all"]:
            slack_query = self.enhanced_salt_query(query)
            slack_response = self.search_slack_messages(slack_query)
            
            if slack_response and slack_response.get('ok'):
                messages = slack_response.get('messages', {}).get('matches', [])
                if messages:
                    contexts = self.extract_thread_contexts(messages)
                    ranked_contexts = self.rank_contexts_by_relevance(query, contexts)
                    
                    for ctx in ranked_contexts[:top_k]:
                        ctx['source_type'] = 'slack'
                        ctx['title'] = f"#{ctx['channel']} - {ctx['user']}"
                        results['slack_results'].append(ctx)
        
        # Documentation search
        if source in ["docs", "all"] and self.doc_search:
            clean_query = self.clean_docs_query(query)
            doc_results = self.doc_search.search_documents(clean_query, top_k)
            results['doc_results'] = doc_results
        
        # Discourse search
        if source in ["discourse", "all"] and self.discourse_search:
            clean_query = self.clean_docs_query(query)  # Use same cleaning as docs
            discourse_results = self.discourse_search.search_discourse(clean_query, top_k)
            results['discourse_results'] = discourse_results
        
        # Combine and rank all results
        all_results = results['slack_results'] + results['doc_results'] + results['discourse_results']
        if all_results:
            combined_ranked = self.rank_mixed_results(query, all_results)
            results['combined_results'] = combined_ranked
        
        return results
    
    def rank_mixed_results(self, query: str, mixed_results: List[Dict]) -> List[Dict]:
        """Rank mixed Slack and doc results by relevance"""
        if not mixed_results:
            return mixed_results
        
        # Use existing ranking logic but on combined text
        texts = [result['combined_text'] for result in mixed_results]
        query_embedding = self.get_embeddings([query])
        result_embeddings = self.get_embeddings(texts)
        
        if len(result_embeddings) == 0 or len(query_embedding) == 0:
            return mixed_results
        
        dimension = result_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(result_embeddings)
        faiss.normalize_L2(query_embedding)
        
        index.add(result_embeddings)
        scores, indices = index.search(query_embedding, len(mixed_results))
        
        ranked_results = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            idx = indices[0][i]
            
            if idx != -1 and idx < len(mixed_results):
                mixed_results[idx]['similarity_score'] = float(score)
                ranked_results.append(mixed_results[idx])
    
        return ranked_results
    
    def generate_unified_response(self, user_query: str, unified_results: Dict) -> str:
        """Generate response using Slack, docs, and discourse results"""
        combined_results = unified_results.get('combined_results', [])[:10]
        
        if not combined_results:
            return "No relevant results found in Slack, documentation, or discourse."
        
        context_text = "\n\n".join([
            f"Source: {result['source_type'].upper()}\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"Content: {result['combined_text'][:500]}..."
            for result in combined_results
        ])
        
        rag_prompt = f"""
        Based on the following information from Slack conversations, documentation, and discourse discussions, 
        please answer the user's question. Clearly indicate which sources you're using.
        
        User Question: {user_query}
        
        Available Information:
        {context_text}
        
        Please provide a comprehensive answer, citing whether information comes from 
        Slack discussions, official documentation, or discourse community posts.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant answering questions about Omni Analytics using both Slack conversations and official documentation."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

def format_rag_response(rag_response: str) -> str:
    return f"### Response\n\n{rag_response}\n"

def render_unified_sources(unified_results: Dict):
    """Show source list for all three sources."""
    combined_results = unified_results.get("combined_results", [])
    slack_count = sum(1 for r in combined_results if r["source_type"] == "slack")
    doc_count = sum(1 for r in combined_results if r["source_type"] == "docs")
    discourse_count = sum(1 for r in combined_results if r["source_type"] == "discourse")

    with st.expander(f"View Sources ({slack_count} Slack, {doc_count} Docs, {discourse_count} Discourse)", expanded=False):
        for i, res in enumerate(combined_results, 1):
            src = res.get("source_type")
            
            if src == "slack":
                ch = res.get("channel", "—")
                link = res.get("link") or res.get("permalink") or res.get("url")
                if link:
                    st.write(f"**{i}.** [#{ch}]({link})")
                else:
                    st.write(f"**{i}.** #{ch}")
                    
            elif src == "docs":
                title = res.get("title", "Untitled")
                if res.get("url"):
                    st.write(f"**{i}.** [Open doc](https://docs.omni.co/{res['url'][3:-3]})")
                else:
                    st.write(f"**{i}.** {title}")
                    
            elif src == "discourse":
                title = res.get("title", "Untitled")
                url = res.get("url", "")
                if url:
                    st.write(f"**{i}.** [Discourse: {title}]({url})")
                else:
                    st.write(f"**{i}.** Discourse: {title}")

def render_message(role, message, assistant_icon_path="assets/blobby.png"):
    cols = st.columns([1, 10])

    if role == "user":
        with cols[0]: st.markdown("You:")
        with cols[1]: st.markdown(message)

    elif role == "assistant":
        with cols[0]: st.image(assistant_icon_path, width=32)
        with cols[1]: st.markdown(message)          # already formatted HTML/MD

    elif role == "sources":                         # 👈 new
        # put the expander back when replaying history
        with cols[1]:
            render_unified_sources(message)

def main():
    # require_login()
    # ───────────────────────────────── 0. KEYS / CLIENTS ─────────────────────────
    slack_token = os.environ.get("SLACK_API_TOKEN") or st.secrets.get("SLACK_API_TOKEN")
    openai_key  = os.environ.get("OPENAI_API_KEY")   or st.secrets.get("OPENAI_API_KEY")
    if not slack_token or not openai_key:
        st.error("Please set SLACK_API_TOKEN and OPENAI_API_KEY.")
        st.stop()

    docs_path   = "temp_docs.json"
    discourse_path = "discourse_embeddings.json"
    rag_system = UnifiedRAGSystem(slack_token, openai_key, docs_path, discourse_path)

    # ───────────────────────────────── 1. STATE BOOTSTRAP ────────────────────────
    if "history_file" not in st.session_state:
        st.session_state.history_file = _load_history(CHAT_LOG_PATH)

    if "session_obj" not in st.session_state:
        st.session_state.session_obj = _new_session()

    if "cache_index" not in st.session_state:
        st.session_state.cache_index = None     # lazy-init when first vec comes in
        st.session_state.cache_turns  = []

    # Initialize messages list for chat persistence
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_raw_queries" not in st.session_state:
        st.session_state.last_raw_queries = {}

    # rebuild FAISS cache once per reload
    if st.session_state.cache_index is None and st.session_state.history_file:
        # find the first turn that *does* have an embedding
        first_turn_with_emb = next(
            (
                t
                for s in st.session_state.history_file
                for t in s["turns"]
                if "query_embedding" in t
            ),
            None,
        )

        if first_turn_with_emb:
            dim = len(first_turn_with_emb["query_embedding"])
            st.session_state.cache_index = faiss.IndexFlatIP(dim)

            for sess in st.session_state.history_file:
                for trn in sess["turns"]:
                    emb = trn.get("query_embedding")
                    if not emb:                         # ⬅️  skip the legacy rows
                        continue
                    vec = np.array([emb], dtype="float32")
                    faiss.normalize_L2(vec)
                    st.session_state.cache_index.add(vec)
                    st.session_state.cache_turns.append(trn)

    # ───────────────────────────────── 2. HEADER INFO ────────────────────────────
    if docs_path and os.path.exists(docs_path) and discourse_path and os.path.exists(discourse_path):
        st.success("✅ Searching Docs + Slack + Discourse.")
        search_source = st.selectbox(
            "Search in:",
            ["all", "slack", "docs", "discourse"],
            format_func=lambda x: {
                "all": "All Sources", 
                "slack": "Slack only", 
                "docs": "Docs only",
                "discourse": "Discourse only"
            }[x],
        )
    else:
        st.info("ℹ️ Slack-only search.")
        search_source = "slack"

    # ───────────────────────────────── 3. RENDER PAST CHAT ───────────────────────
    for m in st.session_state.messages:
        render_message(m["role"], m["content"])

    # ───────────────────────────────── 4. HANDLE ONE TURN ────────────────────────
    if prompt := st.chat_input("Ask anything about Omni"):
        # Add user message to session state FIRST
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # ----- Cmd+Shift+Enter / //nocache detection
        bypass_cache = prompt.endswith("\n")
        prompt = prompt.rstrip("\n")
        if "//nocache" in prompt.lower():
            bypass_cache = True
            prompt = prompt.replace("//nocache", "").strip()

        # ----- check cache (if not bypassed)
        cached_turn = None
        if not bypass_cache and st.session_state.cache_index is not None:
            q_emb = rag_system.get_embeddings([prompt])[0].astype("float32")
            faiss.normalize_L2(q_emb.reshape(1, -1))
            D, I = st.session_state.cache_index.search(q_emb.reshape(1, -1), 1)
            if D[0][0] >= 0.92:
                cached_turn = st.session_state.cache_turns[I[0][0]]

        # ----- Return cached answer if we have one
        if cached_turn:
            # Add cached response to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": format_rag_response(cached_turn["assistant_response"])+ "\n\n*Results from cache*"
            })
            st.session_state.messages.append({
                "role": "sources", 
                "content": {"combined_results": cached_turn["sources"]}
            })
            st.rerun()  # Force rerun to show the messages
            return

        # ───────  fresh search branch  ───────
        slack_query = docs_query = None   # pre-declare so they exist later

        with st.expander("Raw queries", expanded=False):
            if search_source == "slack":
                slack_query = rag_system.enhanced_salt_query(prompt)
                st.markdown(f"**Slack query:** `{slack_query}`")
            elif search_source == "docs":
                docs_query = rag_system.clean_docs_query(prompt)
                st.markdown(f"**Docs query:** `{docs_query}`")
            else:
                slack_query = rag_system.enhanced_salt_query(prompt)
                docs_query  = rag_system.clean_docs_query(prompt)
                st.markdown(f"**Slack query:** `{slack_query}`")
                st.markdown(f"**Docs  query:** `{docs_query}`")

        with st.spinner(f"Searching {search_source}…"):
            unified_results = rag_system.search_unified(prompt, search_source)

            if not unified_results["combined_results"]:
                no_results_msg = f"No relevant results found in {search_source}."
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
                st.rerun()
                return

            rag_response = rag_system.generate_unified_response(prompt, unified_results)
        
        # Add assistant response and sources to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": format_rag_response(rag_response)
        })
        st.session_state.messages.append({
            "role": "sources", 
            "content": unified_results
        })

        # ────────────────────────── 5. LOG TURN & UPDATE CACHE ───────────────────
        query_emb = rag_system.get_embeddings([prompt])[0].tolist()
        turn = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_query": prompt,
            "query_embedding": query_emb,
            "raw_queries": {
                "slack": slack_query if search_source in ("slack", "both") else None,
                "docs":  docs_query  if search_source in ("docs",  "both") else None,
            },
            "assistant_response": rag_response,
            "sources": [
                {
                    "source_type": r["source_type"],
                    "title":      r.get("title"),
                    "link":       r.get("permalink") or r.get("url"),
                    "similarity": r.get("similarity_score", 0),
                }
                for r in unified_results["combined_results"][:10]
            ],
        }

        st.session_state.session_obj["turns"].append(turn)
        if st.session_state.session_obj not in st.session_state.history_file:
            st.session_state.history_file.append(st.session_state.session_obj)
        _save_history(st.session_state.history_file, CHAT_LOG_PATH)

        # ---- update FAISS index
        vec = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(vec)
        if st.session_state.cache_index is None:
            st.session_state.cache_index = faiss.IndexFlatIP(len(query_emb))
        st.session_state.cache_index.add(vec)
        st.session_state.cache_turns.append(turn)

        # Force rerun to display the new messages
        st.rerun()


            
if __name__ == "__main__":
    main()
