import streamlit as st
import os
import json
import numpy as np
import faiss
from datetime import datetime, timezone
from store.config_store import ConfigStore
from store.history_store import HistoryStore
from store.faiss_index_store import FaissIndexStore
from tools.query_router import QueryRouter
from ui.chat_view import (
    render_query_input, 
    render_response_output, 
    render_chat_history,
    render_search_interface
)
from models.data_models import UserQuery, RAGResponse
import logging
from tools.logging_utils import ExtraFormatter

# Constants
CHAT_LOG_PATH = "chat_history.json"
BYPASS_CACHE = True  # Set to True to always bypass cache, False to enable cache

# Update default paths for embeddings
DOCS_JSON_DEFAULT = "models/embeddings/temp_docs.json"
DISCOURSE_JSON_DEFAULT = "models/embeddings/discourse_embeddings.json"

# Logging setup
logger = logging.getLogger("omni_gpt")
logger.setLevel(logging.INFO)
# Only add handlers if not already present
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ExtraFormatter('%(levelname)s %(name)s: %(message)s'))
    logger.addHandler(console_handler)


@st.cache_data
def load_history_cached(path: str) -> list[dict]:
    """Load chat history from JSON file - cached to prevent reloading"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Validate the data structure
        if not isinstance(data, list):
            print(f"Warning: History file {path} is not a list, starting fresh")
            return []
        
        # Filter out invalid sessions
        valid_sessions = []
        for session in data:
            if isinstance(session, dict) and "session_id" in session:
                valid_sessions.append(session)
            else:
                print(f"Warning: Skipping invalid session in history file")
        
        return valid_sessions
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load history from {path}: {e}")
        return []
    except Exception as e:
        print(f"Warning: Unexpected error loading history: {e}")
        # Create backup of corrupted file
        try:
            import shutil
            backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(path, backup_path)
            print(f"Created backup of corrupted file: {backup_path}")
        except Exception as backup_error:
            print(f"Failed to create backup: {backup_error}")
        return []

@st.cache_data
def load_config_cached() -> dict:
    """Load configuration - cached to prevent reloading"""
    config = ConfigStore()
    return {
        "slack_token": config.get("SLACK_TOKEN") or config.get("SLACK_API_TOKEN"),
        "openai_key": config.get("OPENAI_API_KEY"),
        "docs_json": config.get("DOCS_JSON_PATH", DOCS_JSON_DEFAULT),
        "discourse_json": config.get("DISCOURSE_JSON_PATH", DISCOURSE_JSON_DEFAULT),
        "mcp_url": config.get("MCP_URL"),
        "mcp_api_key": config.get("MCP_API_KEY"),
        "mcp_model_id": config.get("MCP_MODEL_ID"),
        "mcp_topic": config.get("MCP_TOPIC_NAME")
    }

@st.cache_resource
def get_query_router_cached(config: dict) -> QueryRouter:
    """Get QueryRouter instance - cached as resource to prevent recreation"""
    faiss_store = FaissIndexStore()
    return QueryRouter(
        slack_token=config["slack_token"],
        openai_api_key=config["openai_key"],
        faiss_store=faiss_store,
        docs_json=config["docs_json"],
        discourse_json=config["discourse_json"],
        mcp_url=config["mcp_url"],
        mcp_api_key=config["mcp_api_key"],
        mcp_model_id=config["mcp_model_id"],
        mcp_topic=config["mcp_topic"]
    )

@st.cache_resource
def get_history_store_cached() -> HistoryStore:
    """Get HistoryStore instance - cached as resource"""
    return HistoryStore(CHAT_LOG_PATH)

def _save_history(history: list[dict], path: str) -> None:
    """Save chat history to JSON file"""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def _new_session() -> dict:
    """Create a new chat session"""
    return {
        "session_id": datetime.now(timezone.utc).isoformat(),
        "turns": []
    }

def main():
    st.set_page_config(page_title="OmniGPT")

    # Load configuration - cached
    config = load_config_cached()
    openai_key = config["openai_key"]
    slack_token = config["slack_token"]
    
    # Validate required configuration
    if not openai_key:
        st.error("❌ OPENAI_API_KEY is required but not found. Please set it in your environment or .env file.")
        st.stop()
    
    if not slack_token:
        st.warning("⚠️ SLACK_TOKEN not found. Slack search will be disabled.")

    use_mcp = all([config["mcp_url"], config["mcp_api_key"], config["mcp_model_id"], config["mcp_topic"]])

    # Initialize stores and router - cached
    history_store = get_history_store_cached()
    router = get_query_router_cached(config)

    # ───────────────────────────────── 1. STATE BOOTSTRAP ────────────────────────
    if "history_file" not in st.session_state:
        st.session_state.history_file = load_history_cached(CHAT_LOG_PATH)

    if "session_obj" not in st.session_state:
        st.session_state.session_obj = _new_session()

    if "cache_index" not in st.session_state:
        st.session_state.cache_index = None     # lazy-init when first vec comes in
        st.session_state.cache_turns = []

    # Initialize messages list for chat persistence
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # rebuild FAISS cache once per reload - only if not already done
    if st.session_state.cache_index is None and st.session_state.history_file:
        # find the first turn that *does* have an embedding
        first_turn_with_emb = None
        for sess in st.session_state.history_file:
            if not isinstance(sess, dict):
                continue
            turns = sess.get("turns", [])
            if not isinstance(turns, list):
                continue
            for t in turns:
                if isinstance(t, dict) and "query_embedding" in t:
                    first_turn_with_emb = t
                    break
            if first_turn_with_emb:
                break

        if first_turn_with_emb:
            dim = len(first_turn_with_emb["query_embedding"])
            st.session_state.cache_index = faiss.IndexFlatIP(dim)

            for sess in st.session_state.history_file:
                if not isinstance(sess, dict):
                    continue
                turns = sess.get("turns", [])
                if not isinstance(turns, list):
                    continue
                for trn in turns:
                    if not isinstance(trn, dict):
                        continue
                    emb = trn.get("query_embedding")
                    if not emb:  # skip the legacy rows
                        continue
                    try:
                        vec = np.array([emb], dtype="float32")
                        faiss.normalize_L2(vec)
                        st.session_state.cache_index.add(vec)
                        st.session_state.cache_turns.append(trn)
                    except Exception as e:
                        print(f"Warning: Failed to add turn to cache: {e}")
                        continue

    # ───────────────────────────────── 2. HEADER INFO ────────────────────────────
    docs_json = config["docs_json"]
    discourse_json = config["discourse_json"]
    if docs_json and os.path.exists(docs_json) and discourse_json and os.path.exists(discourse_json):
        st.success("✅ Searching Docs + Slack + Community. Use \"Hey Blobby\" for MCP")
    else:
        st.info("ℹ️ Slack-only search. Add newline (shift+enter) to run without cache")

    # ───────────────────────────────── 3. RENDER PAST CHAT ───────────────────────
    render_chat_history(st.session_state.messages)

    # ───────────────────────────────── 4. SEARCH INTERFACE ────────────────────────
    search_source = render_search_interface()

    # ───────────────────────────────── 5. HANDLE ONE TURN ────────────────────────
    if prompt := render_query_input():
        # Check if this is a new query (prevent duplicate processing)
        if "last_processed_query" not in st.session_state or st.session_state.last_processed_query != prompt:
            st.session_state.last_processed_query = prompt
            
            # 1. Add user message to session state and render immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Show spinner while processing
            with st.spinner("Searching..."):
                # ----- Cmd+Shift+Enter / //nocache detection
                bypass_cache = BYPASS_CACHE
                prompt = prompt.rstrip("\n")

                user_query = UserQuery.from_text(prompt)
                
                try:
                    decision = router.decide_rag_or_mcp(prompt)
                    st.session_state.messages.append({"role": "routing", "content": f"Routing to {decision.upper()}"})

                    response = router.handle_query(
                        user_query, 
                        search_source=search_source,
                        decision=decision
                    )
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    print(f"Query processing error: {e}")
                    # Create a fallback response
                    response = RAGResponse(
                        user_query=user_query,
                        answer=f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
                        contexts=[],
                        used_cache=False,
                        reranking_info={"routing": "error", "error": str(e)}
                    )
                    st.session_state.messages.append({"role": "routing", "content": "Error occurred - using fallback response"})

                # 3. Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response.answer})
                # Add sources if available
                if response.contexts:
                    sources_data = {
                        "combined_results": [
                            {
                                "source_type": ctx.source,
                                "title": ctx.metadata.get("title", ctx.text[:50] + "..."),
                                "permalink": ctx.metadata.get("permalink", ""),
                                "url": ctx.metadata.get("url", ""),
                                "similarity_score": ctx.score
                            }
                            for ctx in response.contexts
                        ]
                    }
                    st.session_state.messages.append({"role": "sources", "content": sources_data})
                # Generate embedding for caching
                if response.user_query.embedding is None:
                    try:
                        embedding_response = router.openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=[response.user_query.text]
                        )
                        response.user_query.embedding = np.array(embedding_response.data[0].embedding, dtype='float32')
                    except Exception as e:
                        print(f"Warning: Failed to generate embedding: {e}")
                        response.user_query.embedding = np.array([], dtype='float32')
                # Create turn record for history
                turn = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "user_query": prompt,
                    "query_embedding": response.user_query.embedding.tolist() if len(response.user_query.embedding) > 0 else None,
                    "assistant_response": response.answer,
                    "used_cache": response.used_cache,
                    "reranking_info": response.reranking_info,
                    "sources": [
                        {
                            "source_type": ctx.source,
                            "title": ctx.metadata.get("title", ctx.text[:50] + "..."),
                            "link": ctx.metadata.get("permalink") or ctx.metadata.get("url", ""),
                            "similarity": ctx.score,
                        }
                        for ctx in response.contexts[:10]
                    ],
                }
                # Save to session and history
                st.session_state.session_obj["turns"].append(turn)
                if st.session_state.session_obj not in st.session_state.history_file:
                    st.session_state.history_file.append(st.session_state.session_obj)
                _save_history(st.session_state.history_file, CHAT_LOG_PATH)
                # Update FAISS cache
                if len(response.user_query.embedding) > 0:
                    try:
                        vec = np.array([response.user_query.embedding], dtype="float32")
                        faiss.normalize_L2(vec)
                        if st.session_state.cache_index is None:
                            st.session_state.cache_index = faiss.IndexFlatIP(len(response.user_query.embedding))
                        st.session_state.cache_index.add(vec)
                        st.session_state.cache_turns.append(turn)
                    except Exception as e:
                        print(f"Warning: Failed to update cache: {e}")
            
            # Trigger a rerun to show the updated chat
            st.rerun()

if __name__ == "__main__":
    main()