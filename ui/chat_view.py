import streamlit as st
from typing import Dict, List
from models.data_models import RAGResponse
import html

def render_query_input():
    return st.chat_input("Ask anything about Omni")

def render_message(role: str, message, assistant_icon_path="assets/blobby.png"):
    """Render a single message with proper styling"""
    if role == "user":
        safe_message = html.escape(message)
        # User message bubble on the right
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                font-size: 14px;
                line-height: 1.4;
            ">
                {safe_message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        cols = st.columns([1, 10])
        with cols[0]: 
            st.image(assistant_icon_path, width=32)
        with cols[1]: 
            st.markdown(message)  # already formatted HTML/MD

    elif role == "sources":
        # put the expander back when replaying history
        cols = st.columns([1, 10])
        with cols[1]:
            render_unified_sources(message)
    
    elif role == "raw_queries":
        # put the raw queries expander back when replaying history
        cols = st.columns([1, 10])
        with cols[1]:
            render_raw_queries(message)

    elif role == "routing":
        cols = st.columns([1, 10])
        with cols[1]:
            st.markdown(f"*{message}*")

def render_unified_sources(unified_results: Dict):
    """Render sources in an expander"""
    with st.expander("Sources", expanded=False):
        combined_results = unified_results.get("combined_results", [])
        if not combined_results:
            st.info("No sources found")
            return
        
        for i, result in enumerate(combined_results[:10]):
            source_type = result.get("source_type", "unknown")
            title = result.get("title", "No title")
            link = result.get("permalink") or result.get("url", "")
            # Color coding based on source
            color_map = {
                "slack": "#4A154B",
                "docs": "#007C77", 
                "discourse": "#FF6B35"
            }
            color = color_map.get(source_type, "#666")

            # For docs/discourse: title as link; for slack: channel as link
            if source_type == "slack":
                display_title = title if title and title != "No title" and title != "None" else (result.get("channel") or "Slack Channel")
                if link:
                    st.markdown(f"<div style=\"border-left: 4px solid {color}; padding-left: 12px; margin: 8px 0;\"><strong>SLACK</strong>: <a href='{link}' target='_blank'>{display_title}</a></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style=\"border-left: 4px solid {color}; padding-left: 12px; margin: 8px 0;\"><strong>SLACK</strong>: {display_title}</div>", unsafe_allow_html=True)
            else:
                # Fallback: use last part of URL as title if title is missing/None
                display_title = title
                if not title or title == "No title" or title == "None":
                    if link:
                        import urllib.parse
                        last_part = link.rstrip('/').split('/')[-1]
                        display_title = last_part.replace('-', ' ').replace('_', ' ').title()
                    else:
                        display_title = "Untitled"
                if link:
                    st.markdown(f"<div style=\"border-left: 4px solid {color}; padding-left: 12px; margin: 8px 0;\"><strong>{source_type.upper()}</strong>: <a href='{link}' target='_blank'>{display_title}</a></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style=\"border-left: 4px solid {color}; padding-left: 12px; margin: 8px 0;\"><strong>{source_type.upper()}</strong>: {display_title}</div>", unsafe_allow_html=True)

def render_raw_queries(raw_queries: Dict):
    """Show raw queries used for the search"""
    with st.expander("Raw queries", expanded=False):
        if raw_queries.get("slack"):
            st.markdown(f"**Slack query:** `{raw_queries['slack']}`")
        if raw_queries.get("docs"):
            st.markdown(f"**Docs query:** `{raw_queries['docs']}`")
        if raw_queries.get("mcp"):
            st.markdown("**MCP query:** `Direct MCP call`")

def render_response_output(response: RAGResponse):
    """Render the complete response with sources and metadata"""
    # Add routing info if available
    routing_info = response.reranking_info.get("routing", "rag")
    if routing_info == "mcp":
        st.info("🤖 Routed to MCP")
    elif routing_info == "rag":
        search_source = response.reranking_info.get("search_source", "all")
        st.info(f"🔍 Routed to RAG ({search_source})")
    
    # Show cache status
    if response.used_cache:
        st.success("⚡ Response from cache")
    
    # Render the answer
    render_message("assistant", response.answer)
    
    # Render sources if available
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
        render_message("sources", sources_data)

def render_chat_history(messages: List[Dict]):
    """Render the complete chat history"""
    for message in messages:
        render_message(message["role"], message["content"])

def render_search_interface():
    """Render the search source selector"""
    st.markdown("---")
    
    # Search source selector
    if "search_source" not in st.session_state:
        st.session_state.search_source = "all"
    
    search_source = st.selectbox(
        "Search in:",
        ["all", "slack", "docs", "discourse"],
        index=["all", "slack", "docs", "discourse"].index(st.session_state.search_source),
        format_func=lambda x: {
            "all": "All Sources", 
            "slack": "Slack only", 
            "docs": "Docs only",
            "discourse": "Community only"
        }[x],
        key="search_source_selector"
    )
    st.session_state.search_source = search_source
    
    return search_source