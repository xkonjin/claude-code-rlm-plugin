"""
RLM Plugin for Claude Code
Recursive Language Model implementation for processing massive contexts
"""

import os
import json
import hashlib
from typing import Optional, Union, Any, Dict, List
from pathlib import Path

from .context_router import ContextRouter, ContextData
from .repl_engine import RLMREPLEngine
from .agent_manager import ParallelAgentManager
from .integrations.claude_tools import ClaudeToolsIntegration
from .llm_backends import get_llm_manager


DEFAULT_CONFIG = {
    "auto_trigger": {
        "file_size_kb": 50,
        "token_count": 100_000,
        "file_count": 10,
        "enabled": True,
    },
    "processing": {
        "max_concurrent_agents": 8,
        "chunk_overlap_percent": 10,
        "recursion_depth_limit": 2,
        "temp_cleanup": True,
        "chunk_size": 50_000,
    },
    "models": {
        "extraction": "haiku",
        "analysis": "sonnet",
        "orchestration": "sonnet",
    },
}


class RLMPlugin:
    """Main plugin entry point for Claude Code RLM integration"""
    
    def __init__(self):
        self.config = self._load_config()
        self.llm_manager = get_llm_manager()
        
        self.router = ContextRouter(self.config)
        self.repl = RLMREPLEngine(config=self.config)
        model_map = {
            "query": self.config.get("models", {}).get("extraction", "haiku"),
            "extraction": self.config.get("models", {}).get("extraction", "haiku"),
            "analysis": self.config.get("models", {}).get("analysis", "sonnet"),
            "synthesis": self.config.get("models", {}).get("orchestration", "sonnet"),
        }
        self.agent_manager = ParallelAgentManager(
            max_concurrent=self.config['processing']['max_concurrent_agents'],
            model_map=model_map
        )
        self._cache = {}
        
        # Log initialization status
        status = self.llm_manager.get_status()
        print(f"RLM Plugin initialized with {status['current']} backend")
    
    def _load_config(self) -> Dict:
        """Load configuration from plugin.json"""
        config_path = Path(__file__).parent.parent / '.claude-plugin' / 'plugin.json'
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        if not config_path.exists():
            return config
        try:
            with open(config_path) as f:
                plugin_data = json.load(f)
        except Exception:
            return config
        return self._merge_config(config, plugin_data.get('configuration', {}))

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge config dictionaries"""
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._merge_config(base[key], value)
            else:
                base[key] = value
        return base
    
    def should_activate(self, context: Any) -> bool:
        """Check if RLM should be activated for this context"""
        if not self.config.get('auto_trigger', {}).get('enabled', True):
            return False
        
        context_data = ContextData.from_context(context)
        should_activate, _, _ = self.router.should_activate_rlm(context_data)
        return should_activate
    
    def process(
        self, 
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        query: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Main processing entry point"""
        if isinstance(file_path, (list, tuple)):
            return self._process_files(list(file_path), query)
        if file_path:
            return self._process_file(file_path, query)
        elif content:
            return self._process_content(content, query)
        else:
            raise ValueError("Either file_path or content must be provided")
    
    def _process_file(self, file_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Process a file with RLM"""
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}"}
        
        file_hash = self._get_file_hash(file_path)
        
        if file_hash in self._cache and not query:
            return self._cache[file_hash]
        
        context_data = ContextData.from_file(file_path)
        should_activate, strategy, metadata = self.router.should_activate_rlm(context_data)
        
        if not should_activate:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if query:
                return self._direct_query(content, query)
            return {"type": "direct", "content": content}
        
        result = self._execute_strategy(file_path, strategy, metadata, query)
        
        if not query:
            self._cache[file_hash] = result
        
        return result
    
    def _process_content(self, content: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Process content string with RLM"""
        context_data = ContextData.from_content(content)
        should_activate, strategy, metadata = self.router.should_activate_rlm(context_data)
        
        if not should_activate:
            if query:
                return self._direct_query(content, query)
            return {"type": "direct", "content": content}
        
        return self._execute_strategy_on_content(content, strategy, metadata, query)
    
    def _execute_strategy(
        self, 
        file_path: str, 
        strategy: str, 
        metadata: Dict, 
        query: Optional[str]
    ) -> Dict[str, Any]:
        """Execute selected RLM strategy on file"""
        processor = (
            self.router.get_strategy(strategy)
            if strategy else
            self.router.select_strategy(
                data_type=Path(file_path).suffix[1:],
                size=os.path.getsize(file_path)
            )
        )
        
        chunks = processor.decompose(file_path, metadata)
        
        if query:
            results = self.agent_manager.process_chunks_sync(chunks, query)
            aggregated = self.agent_manager.aggregate_results(results, query=query)
            return {
                "type": "rlm_query",
                "strategy": strategy,
                "chunks_processed": len(chunks),
                "synthesis_applied": aggregated.get("synthesis_applied", False),
                "result": aggregated
            }
        else:
            return {
                "type": "rlm_decomposed",
                "strategy": strategy,
                "chunks": len(chunks),
                "metadata": metadata
            }

    def _execute_strategy_on_content(
        self,
        content: str,
        strategy: str,
        metadata: Dict,
        query: Optional[str]
    ) -> Dict[str, Any]:
        """Execute RLM strategy on content string"""
        inferred_type = self._infer_content_type(content)
        processor = (
            self.router.get_strategy(strategy)
            if strategy else
            self.router.select_strategy(
                data_type=inferred_type,
                size=len(content)
            )
        )

        chunks = processor.decompose_content(content, metadata)

        if query:
            results = self.agent_manager.process_chunks_sync(chunks, query)
            aggregated = self.agent_manager.aggregate_results(results, query=query)
            return {
                "type": "rlm_query",
                "strategy": strategy,
                "chunks_processed": len(chunks),
                "synthesis_applied": aggregated.get("synthesis_applied", False),
                "result": aggregated
            }
        else:
            return {
                "type": "rlm_decomposed",
                "strategy": strategy,
                "chunks": len(chunks),
                "metadata": metadata
            }

    def _process_files(self, file_paths: List[str], query: Optional[str]) -> Dict[str, Any]:
        """Process multiple files in one request"""
        existing_files = [f for f in file_paths if os.path.isfile(f)]
        missing = [f for f in file_paths if not os.path.isfile(f)]
        if not existing_files:
            return {"error": "No valid files provided", "missing": missing}

        context_data = ContextData.from_context(existing_files)
        should_activate, strategy, metadata = self.router.should_activate_rlm(context_data)

        if not should_activate:
            contents = {}
            for file_path in existing_files:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    contents[file_path] = f.read()
            if query:
                combined = self._format_multi_file_content(contents)
                return self._direct_query(combined, query, content_type="multi_file")
            return {
                "type": "direct_files",
                "files": list(contents.keys()),
                "contents": contents,
                "missing": missing
            }

        chunks = self._build_chunks_for_files(existing_files, metadata)
        if query:
            results = self.agent_manager.process_chunks_sync(chunks, query)
            aggregated = self.agent_manager.aggregate_results(results, query=query)
            return {
                "type": "rlm_query_files",
                "strategy": strategy,
                "chunks_processed": len(chunks),
                "synthesis_applied": aggregated.get("synthesis_applied", False),
                "result": aggregated,
                "missing": missing
            }
        return {
            "type": "rlm_decomposed_files",
            "strategy": strategy,
            "chunks": len(chunks),
            "metadata": metadata,
            "missing": missing
        }

    def _build_chunks_for_files(self, file_paths: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose multiple files into unified chunk list"""
        chunks: List[Dict[str, Any]] = []
        next_id = 0
        for file_path in file_paths:
            processor = self.router.select_strategy(
                data_type=Path(file_path).suffix[1:].lower(),
                size=os.path.getsize(file_path)
            )
            file_chunks = processor.decompose(file_path, metadata)
            for chunk in file_chunks:
                chunk['id'] = next_id
                next_id += 1
                chunk.setdefault('source_file', file_path)
                chunks.append(chunk)
        return chunks

    def _format_multi_file_content(self, contents: Dict[str, str]) -> str:
        """Create a single content string with file separators"""
        parts = []
        for file_path, content in contents.items():
            parts.append(f"FILE: {file_path}\n{content}")
        return "\n\n".join(parts)

    def _infer_content_type(self, content: str) -> str:
        """Infer content type for strategy selection"""
        snippet = content.strip()[:200]
        if snippet.startswith('{') or snippet.startswith('['):
            return "json"
        if snippet.startswith('<'):
            return "xml"
        if '\n' in content and ',' in content.split('\n')[0]:
            return "csv"
        return "text"

    def _direct_query(
        self,
        content: str,
        query: str,
        content_type: str = "text"
    ) -> Dict[str, Any]:
        """Run a direct LLM query without chunking"""
        prompt = self._build_direct_prompt(content, query)
        model = self.config.get("models", {}).get("analysis", "sonnet")
        response = self.llm_manager.query(prompt, model=model)
        return {
            "type": "direct_query",
            "content_type": content_type,
            "model_used": response.model_used,
            "result": response.content,
            "error": response.error,
        }

    def _build_direct_prompt(self, content: str, query: str) -> str:
        """Build prompt for direct (non-chunked) queries"""
        max_content = 50_000
        if len(content) > max_content:
            keep_edge = 500
            middle_budget = max_content - (keep_edge * 2) - 50
            content = (
                content[:keep_edge + middle_budget]
                + f"\n... [{len(content) - max_content} chars omitted] ...\n"
                + content[-keep_edge:]
            )
        return (
            "Analyze the content below and answer the query.\n\n"
            f"QUERY: {query}\n\n"
            f"CONTENT:\n{content}\n\n"
            "INSTRUCTIONS:\n"
            "- Be concise and specific\n"
            "- Preserve names, numbers, dates, identifiers\n"
            "- If the answer is not present, say so explicitly\n\n"
            "ANSWER:"
        )
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for caching"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def repl_session(self) -> RLMREPLEngine:
        """Get REPL session for interactive processing"""
        return self.repl
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM backend status"""
        return self.llm_manager.get_status()
    
    def clear_cache(self):
        """Clear processing cache"""
        self._cache.clear()


def initialize():
    """Initialize the RLM plugin"""
    return RLMPlugin()


__all__ = ['RLMPlugin', 'initialize']
