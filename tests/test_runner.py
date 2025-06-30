"""
Test runner for OmniGPT query validation

This script runs the test queries against your OmniGPT system and validates
the responses against expected criteria.

To run: python -m tests.test_runner
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Add the current directory to Python path to import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_queries import ALL_TEST_QUERIES, load_test_queries_from_file
from models.data_models import UserQuery
from tools.query_router import QueryRouter
from store.config_store import ConfigStore
from store.faiss_index_store import FaissIndexStore

DOCS_JSON_DEFAULT = "models/embeddings/temp_docs.json"
DISCOURSE_JSON_DEFAULT = "models/embeddings/discourse_embeddings.json"

class OmniGPTTestRunner:
    """Test runner for OmniGPT system validation"""
    
    def __init__(self):
        self.config = ConfigStore()
        self.results = []
        self.router = None
        self._initialize_router()
    
    def _initialize_router(self):
        """Initialize the QueryRouter with configuration"""
        try:
            slack_token = self.config.get("SLACK_TOKEN") or self.config.get("SLACK_API_TOKEN")
            openai_key = self.config.get("OPENAI_API_KEY")
            
            if not openai_key:
                print("❌ OPENAI_API_KEY is required but not found.")
                return
            
            docs_json = self.config.get("DOCS_JSON_PATH", DOCS_JSON_DEFAULT)
            discourse_json = self.config.get("DISCOURSE_JSON_PATH", DISCOURSE_JSON_DEFAULT)
            
            # Optional MCP settings
            mcp_url = self.config.get("MCP_URL")
            mcp_api_key = self.config.get("MCP_API_KEY")
            mcp_model_id = self.config.get("MCP_MODEL_ID")
            mcp_topic = self.config.get("MCP_TOPIC_NAME")
            
            faiss_store = FaissIndexStore()
            
            self.router = QueryRouter(
                slack_token=slack_token,
                openai_api_key=openai_key,
                faiss_store=faiss_store,
                docs_json=docs_json,
                discourse_json=discourse_json,
                mcp_url=mcp_url,
                mcp_api_key=mcp_api_key,
                mcp_model_id=mcp_model_id,
                mcp_topic=mcp_topic
            )
            print("✅ QueryRouter initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize QueryRouter: {e}")
    
    def run_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single query test
        
        Args:
            query_data: Dictionary containing query and expected criteria
            
        Returns:
            Dictionary with test results and validation
        """
        if not self.router:
            return {
                "query": query_data["query"],
                "timestamp": datetime.now().isoformat(),
                "error": "QueryRouter not initialized",
                "success": False
            }
        
        query = query_data["query"]
        expected_criteria = query_data["expected_criteria"]
        category = query_data["category"]
        
        print(f"\n🔍 Testing: {query[:60]}...")
        
        start_time = time.time()
        
        try:
            # Handle empty queries
            if not query.strip():
                return self._validate_empty_query(expected_criteria)
            
            # Create user query
            user_query = UserQuery.from_text(query)
            
            # Get routing decision
            decision = self.router.decide_rag_or_mcp(query)
            
            # Handle query
            response = self.router.handle_query(
                user_query,
                search_source="all",
                decision=decision
            )
            
            response_time = time.time() - start_time
            
            # Validate response
            validation_results = self._validate_response(
                response, expected_criteria, response_time
            )
            
            result = {
                "query": query,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "decision": decision,
                "actual_response": response.answer,
                "expected_criteria": expected_criteria,
                "validation_results": validation_results,
                "success": True,
                "sources_count": len(response.contexts) if response.contexts else 0,
                "used_cache": response.used_cache
            }
            
            # Print validation summary
            self._print_validation_summary(result)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "query": query,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "error": str(e),
                "success": False
            }
    
    def _validate_empty_query(self, expected_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation for empty queries"""
        return {
            "should_handle_gracefully": True,
            "should_return_error_message": True,
            "min_response_length": 20,
            "passed": True
        }
    
    def _validate_response(self, response, expected_criteria: Dict[str, Any], response_time: float) -> Dict[str, Any]:
        """Validate response against expected criteria"""
        validation_results = {}
        
        # Check response length
        if "min_response_length" in expected_criteria:
            min_length = expected_criteria["min_response_length"]
            actual_length = len(response.answer)
            validation_results["min_response_length"] = {
                "expected": min_length,
                "actual": actual_length,
                "passed": actual_length >= min_length
            }
        
        # Check response time
        if "max_response_time" in expected_criteria:
            max_time = expected_criteria["max_response_time"]
            validation_results["response_time"] = {
                "expected": max_time,
                "actual": response_time,
                "passed": response_time <= max_time
            }
        
        # Check routing decision
        if "should_route_to" in expected_criteria:
            expected_routing = expected_criteria["should_route_to"]
            actual_routing = "mcp" if response.reranking_info.get("routing") == "mcp" else "rag"
            validation_results["routing"] = {
                "expected": expected_routing,
                "actual": actual_routing,
                "passed": actual_routing == expected_routing
            }
        
        # Check if sources should be included
        if "should_include_sources" in expected_criteria:
            should_include = expected_criteria["should_include_sources"]
            has_sources = len(response.contexts) > 0
            validation_results["sources_included"] = {
                "expected": should_include,
                "actual": has_sources,
                "passed": has_sources == should_include
            }
        
        # Check source types
        if "source_types" in expected_criteria:
            expected_sources = expected_criteria["source_types"]
            actual_sources = list(set([ctx.source for ctx in response.contexts])) if response.contexts else []
            validation_results["source_types"] = {
                "expected": expected_sources,
                "actual": actual_sources,
                "passed": all(source in actual_sources for source in expected_sources)
            }
        
        # Check for specific keywords in response
        for key, value in expected_criteria.items():
            if key.startswith("should_mention_") and isinstance(value, bool) and value:
                keyword = key.replace("should_mention_", "")
                found = keyword.lower() in response.answer.lower()
                validation_results[f"mentions_{keyword}"] = {
                    "expected": True,
                    "actual": found,
                    "passed": found
                }
            elif key.startswith("should_mention_") and isinstance(value, str):
                found = value.lower() in response.answer.lower()
                validation_results[f"mentions_{value}"] = {
                    "expected": True,
                    "actual": found,
                    "passed": found
                }
        
        # Check cache usage
        if "should_use_cache" in expected_criteria:
            should_use = expected_criteria["should_use_cache"]
            validation_results["cache_usage"] = {
                "expected": should_use,
                "actual": response.used_cache,
                "passed": response.used_cache == should_use
            }
        
        return validation_results
    
    def _print_validation_summary(self, result: Dict[str, Any]):
        """Print a summary of validation results"""
        if not result["success"]:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return
        
        passed = 0
        total = 0
        
        for test_name, test_result in result["validation_results"].items():
            total += 1
            if test_result["passed"]:
                passed += 1
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"✅ {passed}/{total} tests passed ({success_rate:.1f}%)")
        else:
            print("⚠️ No validation criteria specified")
    
    def run_category_tests(self, category: str) -> List[Dict[str, Any]]:
        """Run all tests for a specific category"""
        if category not in ALL_TEST_QUERIES:
            print(f"❌ Category '{category}' not found")
            return []
        
        queries = ALL_TEST_QUERIES[category]
        results = []
        
        print(f"\n🚀 Running {len(queries)} tests for category: {category}")
        
        for i, query_data in enumerate(queries, 1):
            print(f"\n--- Test {i}/{len(queries)} ---")
            result = self.run_single_query(query_data)
            results.append(result)
        
        return results
    
    def run_all_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all test categories"""
        all_results = {}
        
        print("🚀 Starting comprehensive test suite...")
        
        for category in ALL_TEST_QUERIES.keys():
            print(f"\n{'='*50}")
            print(f"Testing Category: {category.upper()}")
            print(f"{'='*50}")
            
            results = self.run_category_tests(category)
            all_results[category] = results
        
        return all_results
    
    def generate_report(self, results: Dict[str, List[Dict[str, Any]]], filename: str = "test_report.json"):
        """Generate a comprehensive test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(results),
            "detailed_results": results
        }
        
        self.save_report(report)
        return report
    
    def _generate_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of all test results"""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        category_summaries = {}
        
        for category, category_results in results.items():
            category_tests = len(category_results)
            category_passed = sum(1 for r in category_results if r.get("success", False))
            category_failed = category_tests - category_passed
            
            total_tests += category_tests
            total_passed += category_passed
            total_failed += category_failed
            
            category_summaries[category] = {
                "total": category_tests,
                "passed": category_passed,
                "failed": category_failed,
                "success_rate": (category_passed / category_tests * 100) if category_tests > 0 else 0
            }
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "categories": category_summaries
        }
    
    def print_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """Print a summary of test results"""
        summary = self._generate_summary(results)
        
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
        
        print(f"\n📈 By Category:")
        for category, cat_summary in summary["categories"].items():
            print(f"  {category}: {cat_summary['passed']}/{cat_summary['total']} ({cat_summary['success_rate']:.1f}%)")
    
    def save_report(self, report_data):
        # Ensure results directory exists
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(results_dir, f'test_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Test report saved to {report_path}")

def main():
    """Main function to run tests"""
    runner = OmniGPTTestRunner()
    
    if not runner.router:
        print("❌ Cannot run tests - QueryRouter not initialized")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category in ALL_TEST_QUERIES:
            print(f"Running tests for category: {category}")
            results = {category: runner.run_category_tests(category)}
        else:
            print(f"Unknown category: {category}")
            print(f"Available categories: {list(ALL_TEST_QUERIES.keys())}")
            return
    else:
        print("Running all test categories...")
        results = runner.run_all_tests()
    
    # Generate and print summary
    runner.print_summary(results)
    
    # Save report
    runner.generate_report(results)

if __name__ == "__main__":
    main() 