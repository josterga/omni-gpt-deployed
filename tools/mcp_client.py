import requests
import uuid
import json

class MCPClient:
    def __init__(self, base_url: str, api_key: str, model_id: str, topic_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.topic_name = topic_name
        self.initialized = False

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-MCP-Model-ID": self.model_id,
            "X-MCP-Topic-Name": self.topic_name
        }

    def _post(self, payload: dict) -> requests.Response:
        return requests.post(self.base_url, headers=self._headers(), json=payload)

    def initialize(self) -> None:
        if self.initialized:
            return
        init_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {},
                "clientInfo": {"name": "omni-gpt", "version": "1.0"}
            }
        }
        self._post(init_payload)
        self.initialized = True

    def run_inference(self, prompt: str) -> str:
        self.initialize()
        # Step 1: Convert NL to Omni query
        gen_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "naturalLanguageToOmniQuery",
                "arguments": {"prompt": prompt, "modelId": self.model_id, "topicName": self.topic_name, "apiKey": self.api_key}
            }
        }
        gen_resp = self._post(gen_payload)
        data = gen_resp.text.split("data: ", 1)[1]
        parsed = json.loads(data)
        omni_query = json.loads(parsed["result"]["content"][0]["text"])["query"]

        # Step 2: Run Omni query
        run_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "runOmniQuery",
                "arguments": {"omniQuery": json.dumps(omni_query), "apiKey": self.api_key}
            }
        }
        run_resp = self._post(run_payload)
        run_data = run_resp.text.split("data: ", 1)[1]
        parsed_run = json.loads(run_data)
        return parsed_run["result"]["content"][0]["text"]