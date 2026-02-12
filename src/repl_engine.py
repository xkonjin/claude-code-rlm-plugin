"""
RLM REPL Engine - Interactive processing environment with context as variables
"""

import os
import sys
import uuid
import tempfile
import traceback
import logging
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import json

from .llm_backends import get_llm_manager


class RLMREPLEngine:
    """Interactive REPL for RLM processing"""
    
    def __init__(self, llm_query_fn: Optional[Callable] = None, config: Optional[Dict] = None):
        self.session_id = str(uuid.uuid4())[:8]
        self.temp_dir = Path(tempfile.gettempdir()) / f"rlm-{self.session_id}"
        self.temp_dir.mkdir(exist_ok=True)
        self._config = config or {}
        
        # Initialize LLM manager
        self.llm_manager = get_llm_manager()
        
        # Use provided function or create one from manager
        if llm_query_fn:
            self._llm_query_fn = llm_query_fn
        else:
            self._llm_query_fn = self.llm_manager.create_query_function()
        
        self.namespace = {
            'context': None,
            'chunks': [],
            'results': [],
            'temp_dir': str(self.temp_dir),
            'llm_query': self._create_llm_query_wrapper(self._llm_query_fn),
            'load_file': self.load_file,
            'load_content': self.load_content,
            'decompose': self._decompose,
            'query_chunk': self._query_chunk,
            'aggregate': self._aggregate,
            'llm_status': self.get_llm_status,
            '__builtins__': __builtins__,
        }
        
        self._recursion_depth = 0
        self._max_recursion = self._config.get('processing', {}).get('recursion_depth_limit', 2)
        
        # Log LLM status
        status = self.llm_manager.get_status()
        logging.info(f"RLM REPL initialized with {status['current']} backend")
    
    def _create_llm_query_wrapper(self, llm_fn: Optional[Callable]) -> Callable:
        """Create wrapper for LLM queries with recursion protection"""
        def wrapper(prompt: str, context: Optional[str] = None, model: str = "haiku") -> str:
            if self._recursion_depth >= self._max_recursion:
                return f"[Recursion limit ({self._max_recursion}) reached]"
            
            self._recursion_depth += 1
            try:
                if llm_fn:
                    if context:
                        full_prompt = f"Context:\n{context[:10000]}\n\nQuery:\n{prompt}"
                    else:
                        full_prompt = prompt
                    
                    result = llm_fn(full_prompt, model)
                    return result
                else:
                    return f"[No LLM backend available - Query: {prompt[:100]}...]"
            finally:
                self._recursion_depth -= 1
        
        return wrapper
    
    def load_file(self, file_path: str, max_size: int = 100_000) -> Dict[str, Any]:
        """Load file into context with automatic decomposition if needed"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        size = file_path.stat().st_size
        
        if size > max_size:
            chunks = self._decompose_file(file_path, max_size)
            self.namespace['context'] = f"[Large file: {len(chunks)} chunks]"
            self.namespace['chunks'] = chunks
            return {
                "status": "decomposed",
                "chunks": len(chunks),
                "total_size": size
            }
        else:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            self.namespace['context'] = content
            return {
                "status": "loaded",
                "size": size
            }
    
    def load_content(self, content: str, max_size: int = 100_000) -> Dict[str, Any]:
        """Load content string into context"""
        if len(content) > max_size:
            chunks = self._decompose_content(content, max_size)
            self.namespace['context'] = f"[Large content: {len(chunks)} chunks]"
            self.namespace['chunks'] = chunks
            return {
                "status": "decomposed",
                "chunks": len(chunks),
                "total_size": len(content)
            }
        else:
            self.namespace['context'] = content
            return {
                "status": "loaded",
                "size": len(content)
            }
    
    def _decompose_file(self, file_path: Path, chunk_size: int) -> list:
        """Decompose large file into chunks"""
        chunks = []
        with open(file_path, 'r') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append({
                    'id': len(chunks),
                    'content': chunk,
                    'size': len(chunk)
                })
        return chunks
    
    def _decompose_content(self, content: str, chunk_size: int) -> list:
        """Decompose large content into chunks"""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            chunks.append({
                'id': len(chunks),
                'content': chunk,
                'size': len(chunk)
            })
        return chunks
    
    def _decompose(self, data: Any, strategy: str = "auto") -> list:
        """Decompose data using specified strategy"""
        if strategy == "auto":
            if isinstance(data, str):
                if data.strip().startswith('{') or data.strip().startswith('['):
                    try:
                        parsed = json.loads(data)
                        return self._decompose_json(parsed)
                    except:
                        pass
                return self._decompose_content(data, 50_000)
            elif isinstance(data, (dict, list)):
                return self._decompose_json(data)
        
        return []
    
    def _decompose_json(self, data: Any) -> list:
        """Decompose JSON data into logical chunks"""
        chunks = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                chunks.append({
                    'id': len(chunks),
                    'key': key,
                    'content': value,
                    'type': 'dict_item'
                })
        elif isinstance(data, list):
            batch_size = max(1, len(data) // 10)
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                chunks.append({
                    'id': len(chunks),
                    'range': f"[{i}:{i+len(batch)}]",
                    'content': batch,
                    'type': 'list_batch'
                })
        
        return chunks
    
    def _query_chunk(self, chunk: Dict, query: str) -> Any:
        """Query a specific chunk"""
        llm_fn = self.namespace['llm_query']
        return llm_fn(query, context=str(chunk.get('content', '')))
    
    def _aggregate(self, results: list, query: str = None) -> Any:
        """Aggregate results from multiple chunks with optional synthesis.

        If a query is provided and multiple string results exist, runs an LLM
        synthesis pass to merge chunk-level findings into a unified answer.
        """
        if not results:
            return None

        if all(isinstance(r, str) for r in results):
            raw = '\n'.join(results)

            # Semantic synthesis when we have a query and real LLM results
            if (query
                and len(results) > 1
                and not all('[Fallback' in r for r in results)):
                llm_fn = self.namespace['llm_query']
                synthesis_prompt = (
                    f"Synthesize these {len(results)} chunk findings into one answer.\n"
                    f"QUERY: {query}\n\nFINDINGS:\n{raw[:50000]}\n\nSYNTHESIZED ANSWER:"
                )
                try:
                    return llm_fn(synthesis_prompt, model="sonnet")
                except Exception:
                    pass  # fall through to raw concatenation

            return raw
        elif all(isinstance(r, dict) for r in results):
            aggregated = {}
            for r in results:
                aggregated.update(r)
            return aggregated
        else:
            return results
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code in RLM namespace"""
        try:
            old_stdout = sys.stdout
            sys.stdout = self._StringIO()
            
            exec(code, self.namespace)
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            return {
                "status": "success",
                "output": output,
                "namespace_keys": list(self.namespace.keys())
            }
        except Exception as e:
            sys.stdout = old_stdout
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def evaluate(self, expression: str) -> Any:
        """Evaluate Python expression and return result"""
        try:
            result = eval(expression, self.namespace)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_variable(self, name: str) -> Any:
        """Get variable from namespace"""
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set variable in namespace"""
        self.namespace[name] = value
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM backend status for debugging"""
        return self.llm_manager.get_status()
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    class _StringIO:
        """Simple StringIO replacement"""
        def __init__(self):
            self.buffer = []
        
        def write(self, s):
            self.buffer.append(str(s))
        
        def getvalue(self):
            return ''.join(self.buffer)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
