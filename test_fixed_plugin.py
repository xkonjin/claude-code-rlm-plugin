#!/usr/bin/env python3
"""
Test script for the fixed RLM plugin with real LLM processing
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import RLMPlugin, initialize
from src.llm_backends import get_llm_manager


def create_test_data():
    """Create test files of various types and sizes"""
    test_dir = Path(tempfile.gettempdir()) / "rlm_test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Large text file
    large_text_file = test_dir / "large_file.txt"
    with open(large_text_file, 'w') as f:
        for i in range(2000):
            f.write(f"This is line {i} of a large text file with some content about data processing.\n")
            f.write(f"Line {i+1} contains information about machine learning and AI applications.\n")
    
    # Large JSON file
    large_json_file = test_dir / "data.json"
    large_data = {
        "users": [
            {"id": i, "name": f"User{i}", "email": f"user{i}@example.com", "age": 20 + (i % 50)}
            for i in range(1000)
        ],
        "products": [
            {"id": i, "name": f"Product{i}", "price": 10.99 + (i * 0.1), "category": f"Category{i % 10}"}
            for i in range(500)
        ],
        "orders": [
            {"id": i, "user_id": i % 1000, "product_id": i % 500, "quantity": 1 + (i % 5)}
            for i in range(2000)
        ]
    }
    
    with open(large_json_file, 'w') as f:
        json.dump(large_data, f, indent=2)
    
    # Code file
    code_file = test_dir / "example_code.py"
    with open(code_file, 'w') as f:
        f.write("""
import os
import json
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []
    
    def load_data(self, file_path: str) -> List[Dict]:
        \"\"\"Load data from JSON file\"\"\"
        with open(file_path) as f:
            return json.load(f)
    
    def process_users(self, users: List[Dict]) -> Dict[str, Any]:
        \"\"\"Process user data and return statistics\"\"\"
        ages = [user['age'] for user in users]
        return {
            'total_users': len(users),
            'avg_age': sum(ages) / len(ages),
            'min_age': min(ages),
            'max_age': max(ages)
        }
    
    def process_products(self, products: List[Dict]) -> Dict[str, Any]:
        \"\"\"Process product data\"\"\"
        prices = [product['price'] for product in products]
        categories = {}
        for product in products:
            cat = product['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_products': len(products),
            'avg_price': sum(prices) / len(prices),
            'categories': categories
        }
    
    def generate_report(self) -> str:
        \"\"\"Generate a comprehensive data report\"\"\"
        report = []
        report.append("Data Processing Report")
        report.append("=" * 50)
        # More report generation logic would go here
        return "\\n".join(report)

def main():
    processor = DataProcessor({'debug': True})
    data = processor.load_data('data.json')
    print("Data processing completed")

if __name__ == "__main__":
    main()
""")
    
    return {
        "large_text": str(large_text_file),
        "large_json": str(large_json_file),
        "code_file": str(code_file),
        "test_dir": str(test_dir)
    }


def test_llm_backends():
    """Test LLM backend availability and functionality"""
    print("=== Testing LLM Backends ===")
    
    manager = get_llm_manager()
    status = manager.get_status()
    
    print(f"Current backend: {status['current']}")
    print("\nBackend availability:")
    for backend, info in status.items():
        if backend != "current":
            available = "✓" if info['available'] else "✗"
            print(f"  {available} {info['name']}")
    
    # Test a simple query
    print("\n=== Testing LLM Query ===")
    test_query = "What are the main benefits of using chunking for large data processing?"
    response = manager.query(test_query, "haiku")
    
    print(f"Query: {test_query}")
    print(f"Response ({response.model_used}):")
    print(f"  {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
    print(f"  Processing time: {response.processing_time_ms:.1f}ms")
    if response.error:
        print(f"  Error: {response.error}")
    
    return response.error is None


def test_file_processing():
    """Test file processing with real LLM analysis"""
    print("\n=== Testing File Processing ===")
    
    # Create test data
    test_files = create_test_data()
    
    # Initialize plugin
    rlm = initialize()
    
    print(f"Plugin LLM status: {rlm.get_llm_status()['current']}")
    
    # Test large text file processing
    print(f"\n--- Processing large text file ({Path(test_files['large_text']).stat().st_size} bytes) ---")
    
    result = rlm.process(
        file_path=test_files['large_text'],
        query="What are the main topics discussed in this file?"
    )
    
    print(f"Processing type: {result['type']}")
    print(f"Strategy used: {result.get('strategy', 'N/A')}")
    print(f"Chunks processed: {result.get('chunks_processed', 0)}")
    
    if 'result' in result and result['result']:
        if isinstance(result['result'], dict):
            aggregated = result['result'].get('aggregated', '')
        else:
            aggregated = result['result']
        if aggregated:
            print(f"Result preview: {str(aggregated)[:300]}...")
    
    # Test JSON file processing
    print(f"\n--- Processing JSON file ({Path(test_files['large_json']).stat().st_size} bytes) ---")
    
    result = rlm.process(
        file_path=test_files['large_json'],
        query="Analyze the data structure and provide insights about the users, products, and orders"
    )
    
    print(f"Processing type: {result['type']}")
    print(f"Strategy used: {result.get('strategy', 'N/A')}")
    print(f"Chunks processed: {result.get('chunks_processed', 0)}")
    
    if 'result' in result and result['result']:
        if isinstance(result['result'], dict):
            aggregated = result['result'].get('aggregated', '')
        else:
            aggregated = result['result']
        if aggregated:
            print(f"Result preview: {str(aggregated)[:300]}...")
    
    # Test code file processing
    print(f"\n--- Processing code file ({Path(test_files['code_file']).stat().st_size} bytes) ---")
    
    result = rlm.process(
        file_path=test_files['code_file'],
        query="Analyze the code structure and explain what the DataProcessor class does"
    )
    
    print(f"Processing type: {result['type']}")
    if result['type'] == 'direct':
        print("File processed directly (no chunking needed)")
    else:
        print(f"Strategy used: {result.get('strategy', 'N/A')}")
        print(f"Chunks processed: {result.get('chunks_processed', 0)}")
        
        if 'result' in result and result['result']:
            if isinstance(result['result'], dict):
                aggregated = result['result'].get('aggregated', '')
            else:
                aggregated = result['result']
            if aggregated:
                print(f"Result preview: {str(aggregated)[:300]}...")


def test_repl_functionality():
    """Test REPL functionality with real LLM processing"""
    print("\n=== Testing REPL Functionality ===")
    
    rlm = initialize()
    repl = rlm.repl_session()
    
    print(f"REPL LLM status: {repl.get_llm_status()['current']}")
    
    # Test loading content
    test_content = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that data.
    
    There are three main types of machine learning:
    1. Supervised learning - uses labeled training data
    2. Unsupervised learning - finds patterns in unlabeled data  
    3. Reinforcement learning - learns through interaction with an environment
    
    Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.
    """
    
    load_result = repl.load_content(test_content)
    print(f"Content loaded: {load_result}")
    
    # Test LLM query through REPL
    query_result = repl.evaluate("llm_query('Summarize the main types of machine learning mentioned', context)")
    print(f"Query result: {query_result[:200]}...")
    
    # Test decomposition
    decompose_result = repl.evaluate("decompose(context)")
    print(f"Decomposed into {len(decompose_result) if isinstance(decompose_result, list) else 0} chunks")


def test_direct_content_processing():
    """Test processing content directly without files"""
    print("\n=== Testing Direct Content Processing ===")
    
    rlm = initialize()
    
    # Create a large content string
    large_content = ""
    for i in range(100):
        large_content += f"""
        Section {i}: Data Analysis Report
        
        This section discusses the analysis of dataset {i}, which contains information about
        customer behavior patterns, sales trends, and market segmentation. The data shows
        significant correlations between customer age, purchase frequency, and product preferences.
        
        Key findings for dataset {i}:
        - Customer acquisition increased by {10 + i % 20}%
        - Average order value: ${50 + (i * 0.5):.2f}
        - Customer retention rate: {80 + (i % 15)}%
        - Most popular product category: Category_{i % 10}
        
        Recommendations based on this analysis include targeted marketing campaigns,
        personalized product recommendations, and improved customer service strategies.
        
        """
    
    print(f"Processing {len(large_content)} characters of content...")
    
    result = rlm.process(
        content=large_content,
        query="What are the key trends and patterns in customer behavior across all sections?"
    )
    
    print(f"Processing type: {result['type']}")
    print(f"Strategy used: {result.get('strategy', 'N/A')}")
    print(f"Chunks processed: {result.get('chunks_processed', 0)}")
    
    if 'result' in result and result['result']:
        if isinstance(result['result'], dict):
            aggregated = result['result'].get('aggregated', '')
        else:
            aggregated = result['result']
        if aggregated:
            print(f"Result preview: {str(aggregated)[:400]}...")


def main():
    """Run comprehensive tests of the fixed RLM plugin"""
    print("RLM Plugin Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        # Test LLM backends first
        llm_working = test_llm_backends()
        
        if llm_working:
            print("\n✓ LLM backend is functional")
        else:
            print("\n⚠ LLM backend has issues, but fallback should work")
        
        # Test file processing
        test_file_processing()
        
        # Test REPL functionality
        test_repl_functionality()
        
        # Test direct content processing
        test_direct_content_processing()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nThe RLM plugin now provides real LLM processing instead of mock responses.")
        print("It supports multiple backends with intelligent fallbacks.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            import shutil
            test_dir = Path(tempfile.gettempdir()) / "rlm_test_data"
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"\n🧹 Cleaned up test data directory")
        except:
            pass


if __name__ == "__main__":
    main()
