from openai import OpenAI
from tools.slack_search import SlackSearch
from tools.doc_search import DocumentSearch
from tools.discourse_search import DiscourseSearch
from tools.reranker import Reranker
from tools.planner import QueryPlanner
from store.faiss_index_store import FaissIndexStore
from tools.mcp_client import MCPClient
from models.data_models import RAGResponse, RetrievedContext, UserQuery
import numpy as np
import faiss

class QueryRouter:
    def __init__(
        self,
        slack_token: str,
        openai_api_key: str,
        faiss_store: FaissIndexStore,
        docs_json: str = "docs.json",
        discourse_json: str = "discourse.json",
        mcp_url: str = None,
        mcp_api_key: str = None,
        mcp_model_id: str = None,
        mcp_topic: str = None,
    ):
        # Core clients
        self.openai = OpenAI(api_key=openai_api_key)
        self.slack = SlackSearch(token=slack_token, openai_client=self.openai)
        self.docs = DocumentSearch(json_path=docs_json, openai_client=self.openai)
        self.discourse = DiscourseSearch(json_path=discourse_json, openai_client=self.openai)
        self.reranker = Reranker(
            user_weights={},
            source_weights={"slack":1.0, "docs":1.0, "discourse":1.0}
        )
        self.planner = QueryPlanner(self.openai)
        self.faiss_store = faiss_store
        
        # Initialize MCP client if configured
        if all([mcp_url, mcp_api_key, mcp_model_id, mcp_topic]):
            self.mcp = MCPClient(
                base_url=mcp_url,
                api_key=mcp_api_key,
                model_id=mcp_model_id,
                topic_name=mcp_topic
            )
        else:
            self.mcp = None

    def decide_rag_or_mcp(self, query: str) -> str:
        """
        Ask OpenAI whether to route this to RAG or send to MCP.
        Returns: "rag" or "mcp"
        """
        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're a routing agent. Given a user's query, decide if it should be handled by:\n"
                            "- 'rag' → if the question is about documentation, internal discussion, or general support (Slack, Docs, Discourse).\n"
                            "- 'mcp' → if the question is about data insights, metrics, summaries, or begins with 'hey blobby', 'ask blobby', etc.\n\n"
                            "Respond with ONLY one word: 'rag' or 'mcp'."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=3,
                temperature=0
            )

            decision_raw = resp.choices[0].message.content.strip().lower()
            print(f"[DEBUG] OpenAI routing raw response: {decision_raw}")
            return decision_raw if decision_raw in {"rag", "mcp"} else "rag"

        except Exception as e:
            print(f"Routing failed: {e}")
            return "rag"

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])

    def check_cache(self, query: str, cache_index=None, cache_turns=None, similarity_threshold=0.92):
        """Check if query exists in cache"""
        if cache_index is None or cache_turns is None:
            return None
            
        try:
            q_emb = self.get_embeddings([query])
            if len(q_emb) == 0:
                return None
                
            q_emb = q_emb[0].astype("float32")
            faiss.normalize_L2(q_emb.reshape(1, -1))
            D, I = cache_index.search(q_emb.reshape(1, -1), 1)
            
            if D[0][0] >= similarity_threshold and I[0][0] < len(cache_turns):
                return cache_turns[I[0][0]]
        except Exception as e:
            print(f"Cache check failed: {e}")
        
        return None

    BYPASS_CACHE = True  # Set to True to always bypass cache, False to enable cache

    def handle_query(self, user_query: UserQuery, search_source: str = "all", cache_index=None, cache_turns=None, decision=None) -> RAGResponse:
        # Hardcoded cache bypass
        bypass_cache = self.BYPASS_CACHE
        query_text = user_query.text.rstrip("\n")
        
        # Check cache if not bypassed
        cached_turn = None
        if not bypass_cache:
            cached_turn = self.check_cache(query_text, cache_index, cache_turns)
        
        if cached_turn:
            # Return cached response
            return RAGResponse(
                user_query=user_query,
                answer=cached_turn.get("assistant_response", "") + "\n\n*Results from cache*",
                contexts=[],
                used_cache=True
            )
        
        # Route to MCP or RAG
        if decision is None:
            decision = self.decide_rag_or_mcp(query_text)
        
        if decision == "mcp" and self.mcp:
            try:
                mcp_response = self.mcp.run_inference(query_text)
                return RAGResponse(
                    user_query=user_query,
                    answer=mcp_response,
                    contexts=[],
                    used_cache=False,
                    reranking_info={"routing": "mcp"}
                )
            except Exception as e:
                print(f"MCP call failed: {e}")
                # Fall back to RAG
        
        # RAG processing
        # Decompose query
        parts = self.planner.plan(query_text)
        combined_contexts = []
        if search_source == "all":
            # Collect per-source results for each part
            per_source_contexts = {"slack": [], "docs": [], "discourse": []}
            for part in parts:
                # Route to each source individually
                slack_contexts = self._route_to_source(part, "slack")
                docs_contexts = self._route_to_source(part, "docs")
                discourse_contexts = self._route_to_source(part, "discourse")
                per_source_contexts["slack"].extend(slack_contexts)
                per_source_contexts["docs"].extend(docs_contexts)
                per_source_contexts["discourse"].extend(discourse_contexts)
            # Take top-N from each source (by score)
            N = 2  # You can tune this number
            selected_contexts = []
            for source, ctxs in per_source_contexts.items():
                # Sort by score descending (if available)
                ctxs_sorted = sorted(ctxs, key=lambda x: x.score, reverse=True)
                selected_contexts.extend(ctxs_sorted[:N])
            # Now rerank the combined pool
            ranked = self.reranker.rerank(selected_contexts)
        else:
            for part in parts:
                contexts = self._route_to_source(part, search_source)
                combined_contexts.extend(contexts)
            ranked = self.reranker.rerank(combined_contexts)
        answer = self._synthesize(query_text, ranked)
        return RAGResponse(
            user_query=user_query, 
            answer=answer, 
            contexts=ranked,
            used_cache=False,
            reranking_info={"routing": "rag", "search_source": search_source}
        )

    def _route_to_source(self, text: str, search_source: str = "all"):
        # Route based on search source
        contexts = []
        
        if search_source in ["slack", "all"]:
            try:
                slack_results = self.slack.search_messages(text)
                for r in slack_results:
                    # Create proper title from channel and user
                    title = f"#{r.get('channel', '')} - {r.get('user', '')}"
                    
                    contexts.append(RetrievedContext(
                        text=r.get('combined_text', r.get('text', '')),
                        score=0.5,  # Default score for Slack results
                        source='slack',
                        metadata={
                            'user': r.get('user', ''),
                            'channel': r.get('channel', ''),
                            'timestamp': r.get('timestamp', ''),
                            'permalink': r.get('permalink', ''),
                            'title': title
                        }
                    ))
            except Exception as e:
                print(f"Error in Slack search: {e}")
        
        if search_source in ["docs", "all"]:
            try:
                doc_results = self.docs.search(text)
                for r in doc_results:
                    contexts.append(RetrievedContext(
                        text=r.get('text', ''),
                        score=r.get('score', 0.0),
                        source='docs',
                        metadata={
                            **r.get('metadata', {}),
                            'url': r.get('url', '')  # Include the constructed URL
                        }
                    ))
            except Exception as e:
                print(f"Error in document search: {e}")
        
        if search_source in ["discourse", "all"]:
            try:
                discourse_results = self.discourse.search(text)
                for r in discourse_results:
                    contexts.append(RetrievedContext(
                        text=r.get('text', ''),
                        score=r.get('score', 0.0),
                        source='discourse',
                        metadata={
                            **r.get('metadata', {}),
                            'url': r.get('url', '')  # Include the discourse URL
                        }
                    ))
            except Exception as e:
                print(f"Error in discourse search: {e}")
        
        return contexts

    def _synthesize(self, query: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant information found."
        
        # Sort contexts by score to prioritize higher-scoring results
        sorted_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        
        # Build context text for the prompt - increased from 500 to 2000 characters
        context_text = "\n\n".join([
            f"Source: {ctx.source.upper()}\nTitle: {ctx.metadata.get('title', 'N/A')}\nContent: {ctx.text[:2000]}..."
            for ctx in sorted_contexts[:8]  # Reduced from 10 to 8 to allow more content per context
        ])
        # print(context_text)
        rag_prompt = f"""
        Based on the following information from Slack conversations, documentation, and discourse discussions, 
        please answer the user's question. Clearly indicate which sources you're using.
        
        User Question: {query}
        
        Available Information:
        {context_text}
        
        Please provide a comprehensive answer, citing whether information comes from 
        Slack discussions, official documentation, or discourse community posts.
        """
        
        resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant answering questions about Omni Analytics using both Slack conversations and official documentation."},
                {"role": "user", "content": rag_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return resp.choices[0].message.content