#!/usr/bin/env python3
"""
Test suite for Claude Code RLM Plugin
"""

import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import RLMPlugin
from src.context_router import ContextData
from src.repl_engine import RLMREPLEngine
from src.agent_manager import ParallelAgentManager, ChunkTask


def test_context_detection():
    """Test automatic context detection"""
    print("Testing context detection...")
    
    plugin = RLMPlugin()
    
    # Test small file
    small_data = ContextData(
        estimated_tokens=1000,
        files=["test.txt"],
        total_size_bytes=4000
    )
    should_activate, strategy, _ = plugin.router.should_activate_rlm(small_data)
    assert not should_activate, "Should not activate for small files"
    
    # Test large file
    large_data = ContextData(
        estimated_tokens=150_000,
        files=["large.json"],
        total_size_bytes=600_000
    )
    should_activate, strategy, _ = plugin.router.should_activate_rlm(large_data)
    assert should_activate, "Should activate for large files"
    assert strategy == "token_chunking", f"Wrong strategy: {strategy}"
    
    print("✓ Context detection working")


def test_file_chunking():
    """Test file chunking strategy"""
    print("Testing file chunking...")
    
    from src.strategies.file_chunking import FileBasedChunking
    
    chunker = FileBasedChunking(chunk_size=100, overlap=10)
    
    # Create test content
    content = "\n".join([f"Line {i}" for i in range(50)])
    chunks = chunker.decompose_content(content, {"chunk_size": 100})
    
    assert len(chunks) > 1, "Should create multiple chunks"
    assert all('content' in c for c in chunks), "Chunks should have content"
    
    print(f"✓ Created {len(chunks)} chunks from test content")


def test_structural_decomposition():
    """Test JSON/CSV structural decomposition"""
    print("Testing structural decomposition...")
    
    from src.strategies.structural_decomp import StructuralDecomposition
    
    decomposer = StructuralDecomposition(max_chunk_size=100)
    
    # Test JSON
    json_data = json.dumps({
        "key1": list(range(100)),
        "key2": {"nested": "value"},
        "key3": "simple"
    })
    
    chunks = decomposer._decompose_json(json_data)
    assert len(chunks) > 0, "Should decompose JSON"
    
    # Test CSV detection
    csv_data = "name,age,city\nJohn,30,NYC\nJane,25,LA"
    chunks = decomposer._decompose_csv(csv_data)
    assert len(chunks) > 0, "Should decompose CSV"
    
    print(f"✓ Structural decomposition working")


def test_xml_decomposition():
    """Test XML decomposition preserves element content"""
    print("Testing XML decomposition...")

    from src.strategies.structural_decomp import StructuralDecomposition

    decomposer = StructuralDecomposition(max_chunk_size=200)
    xml_data = "<root><item>A</item><item>B</item><item>C</item></root>"
    chunks = decomposer._decompose_xml(xml_data)

    assert len(chunks) > 0, "Should decompose XML"
    assert "<item>" in chunks[0]["content"], "XML content should include elements"

    print("✓ XML decomposition working")


def test_time_window_content():
    """Test time-window splitting preserves full window content"""
    print("Testing time window splitting...")

    from src.strategies.time_window import TimeWindowSplitting

    splitter = TimeWindowSplitting(window_size_lines=12)
    lines = [
        f"2026-02-10 12:00:{i:02d} INFO event {i}"
        for i in range(13)
    ]
    chunks = splitter.decompose_content("\n".join(lines), {})

    assert len(chunks) > 0, "Should produce at least one chunk"
    first_chunk_lines = chunks[0]["content"].split("\n")
    assert first_chunk_lines[0] == lines[0], "Chunk should preserve window start"

    print("✓ Time window splitting working")


def test_direct_query_path():
    """Test direct query path for small content"""
    print("Testing direct query path...")

    plugin = RLMPlugin()
    result = plugin.process(content="Small content for direct query", query="Summarize")

    assert result["type"] == "direct_query", "Should use direct query for small content"
    assert "result" in result, "Direct query should return result"

    print("✓ Direct query path working")


def test_multi_file_processing():
    """Test multi-file processing path"""
    print("Testing multi-file processing...")

    plugin = RLMPlugin()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        f1.write("File one content")
        f2.write("File two content")
        file_paths = [f1.name, f2.name]

    try:
        result = plugin.process(file_path=file_paths, query="Summarize both files")
        assert result["type"] == "direct_query", "Small multi-file query should be direct"
        assert "result" in result, "Multi-file direct query should return result"
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.unlink(path)

    print("✓ Multi-file processing working")


def test_repl_engine():
    """Test REPL engine functionality"""
    print("Testing REPL engine...")
    
    repl = RLMREPLEngine()
    
    # Test loading content
    result = repl.load_content("Test content", max_size=1000)
    assert result['status'] == 'loaded', "Should load small content"
    assert repl.namespace['context'] == "Test content", "Content should be in namespace"
    
    # Test code execution
    exec_result = repl.execute("x = 42")
    assert exec_result['status'] == 'success', "Should execute code"
    assert repl.get_variable('x') == 42, "Variable should be set"
    
    # Test expression evaluation
    eval_result = repl.evaluate("x * 2")
    assert eval_result == 84, "Should evaluate expression"
    
    print("✓ REPL engine working")


def test_parallel_processing():
    """Test parallel agent manager"""
    print("Testing parallel processing...")
    
    manager = ParallelAgentManager(max_concurrent=4)
    
    # Create test chunks
    chunks = [
        {"content": f"Chunk {i}", "id": i}
        for i in range(10)
    ]
    
    results = manager.process_chunks_sync(chunks, "Test query")
    
    assert len(results) == 10, "Should process all chunks"
    assert all(r.chunk_id >= 0 for r in results), "Results should have chunk IDs"
    
    # Test aggregation
    aggregated = manager.aggregate_results(results)
    assert 'chunks_processed' in aggregated, "Should have processing stats"
    
    print(f"✓ Processed {len(results)} chunks in parallel")


def test_integration():
    """Test full plugin integration"""
    print("Testing full integration...")
    
    plugin = RLMPlugin()
    
    # Create large test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        large_data = {"data": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        json.dump(large_data, f)
        temp_file = f.name
    
    try:
        # Test processing
        result = plugin.process(file_path=temp_file)
        assert result['type'] in ['direct', 'rlm_decomposed'], "Should process file"
        
        # Test with query
        result = plugin.process(file_path=temp_file, query="Count items")
        assert 'result' in result or 'content' in result, "Should return results"
        
    finally:
        os.unlink(temp_file)
    
    print("✓ Full integration working")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    plugin = RLMPlugin()
    
    # Test missing file
    result = plugin.process(file_path="/nonexistent/file.txt")
    assert 'error' in str(result).lower(), "Should handle missing files"
    
    # Test empty content  
    result = plugin._process_content("", None)
    assert result['type'] == 'direct', "Should handle empty content"
    
    # Test recursion limits
    repl = RLMREPLEngine()
    repl._max_recursion = 1
    
    # Test that recursion is limited
    result = repl.namespace['llm_query']("Query 1")  # depth 1
    assert "[Recursion limit" in str(result) or "LLM" in str(result), "Should work at depth 1"
    
    # The wrapper should prevent deep recursion
    repl._recursion_depth = 1  # Simulate already at max depth
    result = repl.namespace['llm_query']("Query 2")
    assert "Recursion limit" in str(result), "Should enforce recursion limits"
    
    print("✓ Edge cases handled correctly")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Claude Code RLM Plugin Test Suite")
    print("=" * 50)
    
    tests = [
        test_context_detection,
        test_file_chunking,
        test_structural_decomposition,
        test_xml_decomposition,
        test_time_window_content,
        test_direct_query_path,
        test_multi_file_processing,
        test_repl_engine,
        test_parallel_processing,
        test_integration,
        test_edge_cases
    ]
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            return False
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            return False
    
    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
