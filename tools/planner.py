from openai import OpenAI

class QueryPlanner:
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client

    def plan(self, text: str) -> list[str]:
        """
        Break a complex question into independent subquestions. Returns a list of subquery strings.
        Falls back to the original question if parsing fails.
        """
        import json
        prompt = (
            "Return one to three retrieval queries that maximise recall."
            "NEVER drop domain-specific tokens."
            "KEEP multi-word concepts together (e.g. “default_required_access_grants workbook model”)."
            "FLATTEN table, view, or column identifiers to their generic type (e.g. prod.sales_fact_daily → table)."
            "Split only when the question truly contains separate topics (e.g. two companies to compare)."
            "Limit the entire JSON array to ≤ 25 tokens."
            "Output just the queries as a JSON array of strings."
            f"\n\nQuestion: {text}"
        )
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
    