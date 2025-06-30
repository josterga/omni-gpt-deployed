from openai import OpenAI

class QueryPlanner:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def plan(self, text: str, doc_context: str) -> list[str]:
        """
        Break a complex question into independent subquestions. Returns a list of subquery strings.
        Requires doc_context for more informed decomposition.
        """
        import json
        if not doc_context:
            raise ValueError("doc_context is required for query planning.")
        prompt = (
            "You are a query planner. Your task is to break down a complex user query into a small list of independent subqueries, each representing a unique retrievable concept.\n\n"
            "Definitions:\n"
            "- A subquery is a short, search-optimized string (5-10 tokens) representing one concept from the user question.\n"
            "- Subqueries should maximize recall, not precision.\n"
            "- NEVER drop technical tokens (e.g. 'semantic layer', 'required_access_grants').\n"
            "- KEEP multi-word concepts together.\n"
            "- FLATTEN identifiers: table/view names like `prod.sales_fact_daily` → 'table'.\n"
            "- DO NOT include boolean operators, quotes, or join words.\n"
            "- DO NOT repeat information from the reference context.\n\n"
            "Constraints:\n"
            "- Return a flat JSON array of strings.\n"
            "- Limit the entire array to ≤ 25 tokens total.\n"
            "- Return nothing else (no explanation, no headers).\n\n"
            "Examples:\n"
            "Q: What’s the difference between cost modeling in semantic layer vs metrics layer?\n"
            "A: [\"cost modeling semantic layer\", \"cost modeling metrics layer\"]\n\n"
            "Q: Compare default access grants in workbook vs model files.\n"
            "A: [\"default access grants workbook\", \"default access grants model\"]\n\n"
        )
        prompt += f"\n\nReference documentation (do not quote):\n{doc_context[:3000]}"
        prompt += f"\n\nUser Question:\n{text}"
        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            
            # Strip code fences if present
            if content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            elif content.startswith("```json"):
                content = content[7:].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
            
            # Handle responses that start with "json"
            if content.startswith("json"):
                content = content[4:].strip()
            
            # Try to parse JSON
            try:
                subqueries = json.loads(content)
                if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries):
                    print("[DEBUG]Subqueries: " + str(subqueries))
                    return subqueries
            except json.JSONDecodeError as e:
                print(f"Warning: QueryPlanner.plan failed to parse JSON: {e}")
                print(f"Raw response: {content}")
                
        except Exception as e:
            print(f"Warning: QueryPlanner.plan failed: {e}")
            
        # Fallback: return full query as single-element list
        return [text]
    