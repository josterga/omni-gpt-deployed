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
        try:
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
            response = self._post(init_payload)
            if response.status_code == 200:
                self.initialized = True
            else:
                print(f"MCP initialization failed with status {response.status_code}")
        except Exception as e:
            print(f"MCP initialization error: {e}")

    def run_inference(self, prompt: str) -> str:
        try:
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
            if gen_resp.status_code != 200:
                raise Exception(f"MCP query generation failed with status {gen_resp.status_code}")
            
            # Parse response more safely
            try:
                if "data: " in gen_resp.text:
                    data = gen_resp.text.split("data: ", 1)[1]
                    parsed = json.loads(data)
                    if "result" in parsed and "content" in parsed["result"] and len(parsed["result"]["content"]) > 0:
                        content_text = parsed["result"]["content"][0]["text"]
                        try:
                            omni_query_data = json.loads(content_text)
                            if "query" in omni_query_data:
                                omni_query = omni_query_data["query"]
                            else:
                                # This is the actual Omni query format - use the entire object
                                omni_query = omni_query_data
                        except json.JSONDecodeError:
                            # If content is not JSON, use it directly as query
                            omni_query = content_text
                    else:
                        raise Exception("Invalid MCP response structure")
                else:
                    # Try parsing as direct JSON
                    parsed = gen_resp.json()
                    if "result" in parsed and "content" in parsed["result"] and len(parsed["result"]["content"]) > 0:
                        content_text = parsed["result"]["content"][0]["text"]
                        try:
                            omni_query_data = json.loads(content_text)
                            if "query" in omni_query_data:
                                omni_query = omni_query_data["query"]
                            else:
                                # This is the actual Omni query format - use the entire object
                                omni_query = omni_query_data
                        except json.JSONDecodeError:
                            # If content is not JSON, use it directly as query
                            omni_query = content_text
                    else:
                        raise Exception("Invalid MCP response structure")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse MCP response: {e}")

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
            if run_resp.status_code != 200:
                raise Exception(f"MCP query execution failed with status {run_resp.status_code}")
            
            # Parse response more safely
            try:
                if "data: " in run_resp.text:
                    run_data = run_resp.text.split("data: ", 1)[1]
                    parsed_run = json.loads(run_data)
                else:
                    parsed_run = run_resp.json()
                
                if "result" in parsed_run and "content" in parsed_run["result"] and len(parsed_run["result"]["content"]) > 0:
                    return parsed_run["result"]["content"][0]["text"]
                else:
                    raise Exception("Invalid MCP execution response structure")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse MCP execution response: {e}")
                
        except Exception as e:
            print(f"MCP inference failed: {e}")
            # Return a fallback response
            return f"I apologize, but I encountered an error while processing your request. The error was: {str(e)}. Please try rephrasing your question or contact support if the issue persists."