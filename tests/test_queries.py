"""
Test queries for OmniGPT RAG system validation

This file contains a comprehensive set of test queries that can be used to validate
the response quality and consistency of the OmniGPT system. The queries are organized
by category and include expected behaviors and validation criteria.

Usage:
1. Edit the queries below to match your specific use cases
2. Run the queries against your system
3. Validate responses against the expected criteria
4. Use the results to improve system performance
"""

import json
from datetime import datetime
from typing import List, Dict, Any

class QueryTestSuite:
    """Test suite for validating OmniGPT query responses"""
    
    def __init__(self):
        self.test_results = []
    
    def run_query_test(self, query: str, expected_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single query test and return results
        
        Args:
            query: The test query to run
            expected_criteria: Dictionary of expected response criteria
            
        Returns:
            Dictionary with test results and validation
        """
        # This would be implemented to actually call your system
        # For now, this is a template for manual testing
        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "expected_criteria": expected_criteria,
            "actual_response": None,  # To be filled by actual system call
            "validation_results": {}
        }

# =============================================================================
# SLACK-SPECIFIC QUERIES
# =============================================================================

SLACK_QUERIES = [
    {
        "query": "how to reference period over period fields in a drill query?",
        "category": "slack_drill",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack"],
            "min_response_length": 50,
            "should_mention_flatten_pivot": True
        }
    },
    {
        "query": "What's our recommendation for setting up customers who have users in multiple regions, including EU?",
        "category": "slack_regions",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack"],
            "min_response_length": 100,
            "should_mention_multiple_instances": True
        }
    },
    {
        "query": "What users get schedule failure notifications?",
        "category": "slack_schedule",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack"],
            "min_response_length": 80,
            "should_mention_creator": True,
            "should_mention_email": True
        }
    },
    {
        "query": "What IPs can I whitelist for my webhook?",
        "category": "slack_webhook",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack"],
            "min_response_length": 60,
            "should_mention_connection": True,
            "should_mention_allowlist": True
        }
    }
]

# =============================================================================
# DOCUMENTATION QUERIES
# =============================================================================

DOCS_QUERIES = [
    {
        "query": "How do I use default_filters in a topic?",
        "category": "docs_modeling",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["docs"],
            "min_response_length": 150,
            "should_mention_topic": True,
            "should_mention_default_filter": True,
            "should_mention_model": True
        }
    },
    {
        "query": "does accessboost apply to workbooks or only dashboard?",
        "category": "docs_access",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["docs"],
            "min_response_length": 200,
            "should_mention_accessboost": True,
            "should_mention_workbook": True,
            "should_mention_dashboard": True
        }
    },
    {
        "query": "What are query models and how are they used?",
        "category": "docs_modeling",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["docs"],
            "min_response_length": 300,
            "should_mention_query_model": True,
            "should_mention_sql": True,
            "should_mention_data": True
        }
    },
    {
        "query": "Can a filter apply to multiple charts?",
        "category": "docs_filters",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["docs"],
            "min_response_length": 120,
            "should_mention_filter": True,
            "should_mention_map": True
        }
    }
]

# =============================================================================
# DISCOURSE/COMMUNITY QUERIES
# =============================================================================

DISCOURSE_QUERIES = [
    {
        "query": "How could I share a dashboard from Omni to Google Sheets?",
        "category": "discourse_community",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["discourse"],
            "min_response_length": 100,
            "should_mention_google_cloud_function": True,
            "should_mention_webhook": True
        }
    },
    {
        "query": "How can I filter on a dimension I'm pivoting on",
        "category": "discourse_filter",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["discourse"],
            "min_response_length": 120,
            "should_mention_filter_by_query": True,
        }
    },
    {
        "query": "What is the square icon field picker?",
        "category": "discourse_troubleshooting",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["discourse"],
            "min_response_length": 150,
            "should_mention_unknown": True,
            "should_mention_cast": True
        }
    }
]

# =============================================================================
# MCP (MODEL CONTEXT PROTOCOL) QUERIES
# =============================================================================

MCP_QUERIES = [
    {
        "query": "Hey Blobby, what are our current user engagement metrics?",
        "category": "mcp_metrics",
        "expected_criteria": {
            "should_route_to": "mcp",
            "should_include_sources": False,
            "source_types": [],
            "min_response_length": 100,
            "should_mention_metrics": True,
            "should_mention_engagement": True
        }
    },
    {
        "query": "Ask Blobby for a summary of last month's performance data",
        "category": "mcp_summary",
        "expected_criteria": {
            "should_route_to": "mcp",
            "should_include_sources": False,
            "source_types": [],
            "min_response_length": 200,
            "should_mention_summary": True,
            "should_mention_performance": True
        }
    },
    {
        "query": "Hey Blobby, analyze our conversion rates over the past quarter",
        "category": "mcp_analysis",
        "expected_criteria": {
            "should_route_to": "mcp",
            "should_include_sources": False,
            "source_types": [],
            "min_response_length": 150,
            "should_mention_analysis": True,
            "should_mention_conversion": True
        }
    },
    {
        "query": "What insights can Blobby provide about our user retention?",
        "category": "mcp_insights",
        "expected_criteria": {
            "should_route_to": "mcp",
            "should_include_sources": False,
            "source_types": [],
            "min_response_length": 120,
            "should_mention_insights": True,
            "should_mention_retention": True
        }
    }
]

# =============================================================================
# CROSS-SOURCE QUERIES (should search multiple sources)
# =============================================================================

CROSS_SOURCE_QUERIES = [
    {
        "query": "What's the latest on the authentication system?",
        "category": "cross_source_general",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack", "docs", "discourse"],
            "min_response_length": 200,
            "should_mention_authentication": True,
            "should_mention_system": True
        }
    },
    {
        "query": "How are we handling the recent security concerns?",
        "category": "cross_source_security",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack", "docs", "discourse"],
            "min_response_length": 180,
            "should_mention_security": True,
            "should_mention_concerns": True
        }
    },
    {
        "query": "What's the status of the new feature rollout?",
        "category": "cross_source_status",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "source_types": ["slack", "docs", "discourse"],
            "min_response_length": 160,
            "should_mention_feature": True,
            "should_mention_rollout": True
        }
    }
]

# =============================================================================
# EDGE CASES AND ERROR SCENARIOS
# =============================================================================

EDGE_CASE_QUERIES = [
    {
        "query": "",  # Empty query
        "category": "edge_case_empty",
        "expected_criteria": {
            "should_handle_gracefully": True,
            "should_return_error_message": True,
            "min_response_length": 20
        }
    },
    {
        "query": "?",  # Very short query
        "category": "edge_case_short",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_ask_for_clarification": True,
            "min_response_length": 30
        }
    },
    {
        "query": "This is a very long query that contains many words and should test how the system handles queries that are significantly longer than typical user queries and might contain redundant information or unnecessary details that could potentially confuse the routing or search algorithms",
        "category": "edge_case_long",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "min_response_length": 100,
            "should_handle_long_query": True
        }
    },
    {
        "query": "What about the thing with the stuff and the other thing?",
        "category": "edge_case_vague",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_ask_for_clarification": True,
            "min_response_length": 50,
            "should_mention_vague": True
        }
    },
    {
        "query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
        "category": "edge_case_sql_injection",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_handle_safely": True,
            "should_not_execute_sql": True,
            "min_response_length": 40
        }
    }
]

# =============================================================================
# PERFORMANCE AND CACHE TESTING
# =============================================================================

PERFORMANCE_QUERIES = [
    {
        "query": "What is the current status of our API?",
        "category": "performance_cache_test",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_include_sources": True,
            "max_response_time": 5.0,  # seconds
            "min_response_length": 80,
            "should_mention_api": True,
            "should_mention_status": True
        }
    },
    {
        "query": "What is the current status of our API?",  # Duplicate for cache testing
        "category": "performance_cache_test",
        "expected_criteria": {
            "should_route_to": "rag",
            "should_use_cache": True,
            "max_response_time": 1.0,  # Should be faster with cache
            "min_response_length": 80
        }
    }
]

# =============================================================================
# ALL QUERIES COMBINED
# =============================================================================

ALL_TEST_QUERIES = {
    "slack": SLACK_QUERIES,
    "docs": DOCS_QUERIES,
    "discourse": DISCOURSE_QUERIES
    # "mcp": MCP_QUERIES,
    # "cross_source": CROSS_SOURCE_QUERIES,
    # "edge_cases": EDGE_CASE_QUERIES,
    # "performance": PERFORMANCE_QUERIES
}

def save_test_queries_to_file(filename: str = "test_queries.json"):
    """Save all test queries to a JSON file for easy access"""
    with open(filename, 'w') as f:
        json.dump(ALL_TEST_QUERIES, f, indent=2)
    print(f"Test queries saved to {filename}")

def load_test_queries_from_file(filename: str = "test_queries.json") -> Dict:
    """Load test queries from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def print_query_summary():
    """Print a summary of all available test queries"""
    total_queries = sum(len(queries) for queries in ALL_TEST_QUERIES.values())
    print(f"\n=== OmniGPT Test Query Summary ===")
    print(f"Total queries: {total_queries}")
    print("\nCategories:")
    for category, queries in ALL_TEST_QUERIES.items():
        print(f"  {category}: {len(queries)} queries")
    
    print("\nSample queries by category:")
    for category, queries in ALL_TEST_QUERIES.items():
        if queries:
            print(f"\n{category.upper()}:")
            for i, query_data in enumerate(queries[:2]):  # Show first 2 from each category
                print(f"  {i+1}. {query_data['query']}")

if __name__ == "__main__":
    # Save queries to file
    save_test_queries_to_file()
    
    # Print summary
    print_query_summary()
    
    print("\n=== Usage Instructions ===")
    print("1. Edit the queries in this file to match your specific use cases")
    print("2. Run queries against your OmniGPT system")
    print("3. Validate responses against the expected criteria")
    print("4. Use the results to improve system performance")
    print("\nYou can also load queries from the JSON file:")
    print("queries = load_test_queries_from_file()") 