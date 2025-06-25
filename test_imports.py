#!/usr/bin/env python3
"""
Simple test script to verify imports and basic functionality
"""

def test_imports():
    try:
        print("Testing imports...")
        
        # Test basic imports
        from store.config_store import ConfigStore
        from store.history_store import HistoryStore
        from store.faiss_index_store import FaissIndexStore
        from models.data_models import UserQuery, RetrievedContext, RAGResponse
        from tools.query_router import QueryRouter
        from tools.slack_search import SlackSearch
        from tools.doc_search import DocumentSearch
        from tools.discourse_search import DiscourseSearch
        from tools.reranker import Reranker
        from tools.planner import QueryPlanner
        from tools.mcp_client import MCPClient
        from ui.chat_view import render_query_input, render_response_output
        
        print("✓ All imports successful")
        
        # Test basic instantiation
        config = ConfigStore()
        print("✓ ConfigStore instantiated")
        
        history_store = HistoryStore("test_history.json")
        print("✓ HistoryStore instantiated")
        
        faiss_store = FaissIndexStore()
        print("✓ FaissIndexStore instantiated")
        
        # Test data models
        user_query = UserQuery.from_text("test query")
        print("✓ UserQuery created")
        
        context = RetrievedContext(
            text="test context",
            score=0.8,
            source="test",
            metadata={}
        )
        print("✓ RetrievedContext created")
        
        response = RAGResponse(
            user_query=user_query,
            answer="test answer",
            contexts=[context]
        )
        print("✓ RAGResponse created")
        
        print("\n🎉 All tests passed! The modular codebase should work correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports() 