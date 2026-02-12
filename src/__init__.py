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


class RLMPlugin:
    """Main plugin entry point for Claude Code RLM integration"""
    
    def __init__(self):
        self.config = self._load_config()
        self.llm_manager = get_llm_manager()
        
        self.router = ContextRouter(self.config)
        self.repl = RLMREPLEngine()
        self.agent_manager = ParallelAgentManager(
            max_concurrent=self.config['processing']['max_concurrent_agents']
        )
        self._cache = {}
        
        # Log initialization status
        status = self.llm_manager.get_status()
        print(f"RLM Plugin initialized with {status['current']} backend")
    
    def _load_config(self) -> Dict:
        """Load configuration from plugin.json"""
        config_path = Path(__file__).parent.parent / '.claude-plugin' / 'plugin.json'
        with open(config_path) as f:
            plugin_data = json.load(f)
        return plugin_data.get('configuration', {})
    
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
        if file_path:
            return self._process_file(file_path, query)
        elif content:
            return self._process_content(content, query)
        else:
            raise ValueError("Either file_path or content must be provided")
    
    def _process_file(self, file_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Process a file with RLM"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        file_hash = self._get_file_hash(file_path)
        
        if file_hash in self._cache and not query:
            return self._cache[file_hash]
        
        context_data = ContextData.from_file(file_path)
        should_activate, strategy, metadata = self.router.should_activate_rlm(context_data)
        
        if not should_activate:
            with open(file_path) as f:
                content = f.read()
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
        processor = self.router.select_strategy(
            data_type=Path(file_path).suffix[1:],
            size=os.path.getsize(file_path)
        )
        
        chunks = processor.decompose(file_path, metadata)
        
        if query:
            results = self.agent_manager.process_chunks_sync(chunks, query)
            aggregated = self.agent_manager.aggregate_results(results)
            return {
                "type": "rlm_query",
                "strategy": strategy,
                "chunks_processed": len(chunks),
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
        processor = self.router.select_strategy(
            data_type="text",
            size=len(content)
        )
        
        chunks = processor.decompose_content(content, metadata)
        
        if query:
            results = self.agent_manager.process_chunks_sync(chunks, query)
            aggregated = self.agent_manager.aggregate_results(results)
            return {
                "type": "rlm_query",
                "strategy": strategy,
                "chunks_processed": len(chunks),
                "result": aggregated
            }
        else:
            return {
                "type": "rlm_decomposed",
                "strategy": strategy,
                "chunks": len(chunks),
                "metadata": metadata
            }
    
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