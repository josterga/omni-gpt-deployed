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
import tiktoken
import logging
from store.session_utils import get_current_session_id

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
        self.logger = logging.getLogger("omni_gpt.query_router")
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
            self.logger.info("Routing decision", extra={"event_type": "ROUTING", "details": {"query": query, "decision": decision_raw, "session_id": get_current_session_id()}})
            return decision_raw if decision_raw in {"rag", "mcp"} else "rag"

        except Exception as e:
            self.logger.error("Routing failed", extra={"event_type": "ROUTING_ERROR", "details": {"query": query, "error": str(e), "session_id": get_current_session_id()}})
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
        bypass_cache = self.BYPASS_CACHE
        query_text = user_query.text.rstrip("\n")
        self.logger.info("User query input", extra={"event_type": "USER_QUERY_INPUT", "details": {"query": query_text}, "session_id": get_current_session_id()})
        
        # Check cache if not bypassed
        cached_turn = None
        if not bypass_cache:
            cached_turn = self.check_cache(query_text, cache_index, cache_turns)
        
        if cached_turn:
            self.logger.info("Cache hit", extra={"event_type": "CACHE_HIT", "details": {"query": user_query.text, "session_id": get_current_session_id()}})
            return RAGResponse(
                user_query=user_query,
                answer=cached_turn.get("assistant_response", "") + "\n\n*Results from cache*",
                contexts=[],
                used_cache=True
            )
        
        # Route to MCP or RAG
        if decision is None:
            decision = self.decide_rag_or_mcp(query_text)
        
        self.logger.info("Planned routing decision", extra={"event_type": "PLANNED_ROUTING", "details": {"decision": decision, "query": query_text}, "session_id": get_current_session_id()})
        if decision == "mcp" and self.mcp:
            try:
                mcp_response = self.mcp.run_inference(query_text)
                self.logger.info("MCP call success", extra={"event_type": "MCP_CALL", "details": {"query": query_text, "result": mcp_response}, "session_id": get_current_session_id()})
                return RAGResponse(
                    user_query=user_query,
                    answer=mcp_response,
                    contexts=[],
                    used_cache=False,
                    reranking_info={"routing": "mcp"}
                )
            except Exception as e:
                self.logger.error("MCP call failed", extra={"event_type": "MCP_ERROR", "details": {"query": query_text, "error": str(e)}, "session_id": get_current_session_id()})
                # Fall back to RAG
                decision = "rag"
        
        # RAG processing
        # Decompose query
        # Get top-3 relevant doc context for planner
        doc_results = self.docs.search(query_text, top_k=3)
        doc_context = "\n\n".join([r.get('text', '') for r in doc_results]) if doc_results else None
        parts = self.planner.plan(query_text, doc_context=doc_context)
        self.logger.info("Planned queries", extra={"event_type": "PLANNED_QUERIES", "details": {"parts": parts}, "session_id": get_current_session_id()})
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
                self.logger.info("Source results", extra={"event_type": "SOURCE_RESULTS", "details": {"part": part, "slack": [ctx.text for ctx in slack_contexts], "docs": [ctx.text for ctx in docs_contexts], "discourse": [ctx.text for ctx in discourse_contexts]}, "session_id": get_current_session_id()})
            # Take top-N from each source (by score)
            N = 2  # You can tune this number
            selected_contexts = []
            for source, ctxs in per_source_contexts.items():
                # Sort by score descending (if available)
                if ctxs:
                    ctxs_sorted = sorted(ctxs, key=lambda x: getattr(x, 'score', 0.0), reverse=True)
                    selected_contexts.extend(ctxs_sorted[:N])
            # Now rerank the combined pool
            try:
                ranked = self.reranker.rerank(selected_contexts)
            except Exception as e:
                print(f"Ranking failed: {e}")
                ranked = selected_contexts  # Fallback to unranked
        else:
            for part in parts:
                contexts = self._route_to_source(part, search_source)
                combined_contexts.extend(contexts)
                self.logger.info("Source results", extra={"event_type": "SOURCE_RESULTS", "details": {"part": part, search_source: [ctx.text for ctx in contexts]}, "session_id": get_current_session_id()})
            try:
                ranked = self.reranker.rerank(combined_contexts)
            except Exception as e:
                print(f"Ranking failed: {e}")
                ranked = combined_contexts  # Fallback to unranked
        
        self.logger.info("Ranked contexts", extra={"event_type": "RANKED_CONTEXTS", "details": {"ranked": [ctx.text for ctx in ranked]}, "session_id": get_current_session_id()})
        answer = self._synthesize(query_text, ranked)
        self.logger.info("Final result", extra={"event_type": "FINAL_RESULT", "details": {"answer": answer}, "session_id": get_current_session_id()})
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

        # Sort by score descending
        sorted_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)

        # Tokenizer for GPT-4 family
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        max_tokens_per_context = 512
        context_snippets = []

        for ctx in sorted_contexts:
            tokens = tokenizer.encode(ctx.text)
            truncated_text = tokenizer.decode(tokens[:max_tokens_per_context])
            context_snippets.append(
                f"Source: {ctx.source.upper()}\nTitle: {ctx.metadata.get('title', 'N/A')}\nContent:\n{truncated_text.strip()}"
            )
            if len(context_snippets) >= 8:
                break

        context_text = "\n\n---\n\n".join(context_snippets)

        rag_prompt = f"""
    You are an AI assistant for Omni Analytics. You are answering the user's question using **only the provided information**, which includes Slack conversations, official documentation, and community (discourse) discussions.

    **Important Instructions:**
    - Use only the provided context. **Do not hallucinate** or invent facts.
    - Clearly cite where each point comes from (e.g., "Slack", "Documentation", "Community").
    - Follow the answer structure below.

    ---

    **User Question**:
    {query}

    ---

    **Available Information**:
    {context_text}

    ---

    **Answer Format**:

    1. **Answer**  
    - Summarize the correct response in a clear, human-readable way.  
    - Use only the information from the provided context.  
    - Do **not** hallucinate or add unstated assumptions.
    - If the question or context involves structured data (e.g., YAML, JSON, config files, code), include a **generic example** formatted in a fenced code block.

    2. **Source Highlights**  
    - List key facts or data points from the sources that directly support the answer.  
    - Do not restate entire paragraphs.

    3. **Unanswered Questions** *(if applicable)*  
    - Note any aspects of the user's question that the provided information does **not** answer.  
    - Be concise but honest about the gap.
    - If there are no unanswered questions, don't include this section as part of your answer.

    Now write your answer.
    """

        resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant answering questions about Omni Analytics using retrieved context."},
                {"role": "user", "content": rag_prompt}
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        return resp.choices[0].message.content