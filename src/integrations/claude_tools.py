"""
Claude Code tools integration for RLM plugin
"""

from typing import Union, Optional, Dict, List, Any, Callable
import os
import hashlib
import logging

from ..llm_backends import get_llm_manager


class ClaudeToolsIntegration:
    """Integration with Claude Code's built-in tools"""
    
    def __init__(self, claude_tools=None, rlm_plugin=None):
        self.tools = claude_tools
        self.rlm = rlm_plugin
        self.llm_manager = get_llm_manager()
        self._cache = {}
    
    def smart_read(self, file_path: str) -> Union[str, Dict[str, Any]]:
        """Enhanced Read with automatic RLM activation"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        size = os.path.getsize(file_path)
        
        if size > 50_000 and self.rlm:
            return self.rlm.process(file_path=file_path)
        elif self.tools:
            return self.tools.read(file_path=file_path)
        else:
            with open(file_path) as f:
                return f.read()
    
    def smart_grep(
        self, 
        pattern: str, 
        path: str = ".",
        output_mode: str = "files_with_matches"
    ) -> Union[List[str], Dict[str, Any]]:
        """Enhanced Grep with parallel processing for large codebases"""
        file_count = self._estimate_file_count(path)
        
        if file_count > 1000 and self.rlm:
            return self._rlm_grep(pattern, path, output_mode)
        elif self.tools:
            return self.tools.grep(
                pattern=pattern,
                path=path,
                output_mode=output_mode
            )
        else:
            import subprocess
            result = subprocess.run(
                ["grep", "-r", pattern, path],
                capture_output=True,
                text=True
            )
            return result.stdout.split('\n') if result.stdout else []
    
    def smart_glob(self, pattern: str) -> Union[List[str], Dict[str, Any]]:
        """Enhanced Glob with batching for large result sets"""
        if self.tools:
            matches = self.tools.glob(pattern=pattern)
        else:
            import glob
            matches = glob.glob(pattern, recursive=True)
        
        if len(matches) > 100 and self.rlm:
            return self._rlm_batch_process(matches)
        else:
            return matches
    
    def _estimate_file_count(self, path: str) -> int:
        """Estimate number of files in directory"""
        count = 0
        try:
            for root, dirs, files in os.walk(path):
                count += len(files)
                if count > 1000:
                    break
        except:
            pass
        return count
    
    def _rlm_grep(
        self, 
        pattern: str, 
        path: str, 
        output_mode: str
    ) -> Dict[str, Any]:
        """Parallel grep using RLM"""
        import glob
        files = []
        for ext in ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.go']:
            files.extend(glob.glob(f"{path}/**/{ext}", recursive=True))
        
        if not files:
            return {"matches": [], "files_searched": 0}
        
        file_batches = [files[i:i+100] for i in range(0, len(files), 100)]
        
        results = []
        for batch in file_batches:
            batch_results = self._search_batch(batch, pattern, output_mode)
            results.extend(batch_results)
        
        return {
            "matches": results,
            "files_searched": len(files),
            "pattern": pattern
        }
    
    def _search_batch(
        self, 
        files: List[str], 
        pattern: str, 
        output_mode: str
    ) -> List[Any]:
        """Search a batch of files"""
        import re
        results = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if output_mode == "files_with_matches":
                    if re.search(pattern, content):
                        results.append(file_path)
                elif output_mode == "content":
                    for i, line in enumerate(content.split('\n'), 1):
                        if re.search(pattern, line):
                            results.append(f"{file_path}:{i}:{line}")
                elif output_mode == "count":
                    matches = len(re.findall(pattern, content))
                    if matches:
                        results.append({"file": file_path, "count": matches})
            except:
                continue
        
        return results
    
    def _rlm_batch_process(self, files: List[str]) -> Dict[str, Any]:
        """Process large file lists with RLM"""
        categorized = {}
        
        for file_path in files:
            ext = os.path.splitext(file_path)[1]
            if ext not in categorized:
                categorized[ext] = []
            categorized[ext].append(file_path)
        
        return {
            "total_files": len(files),
            "by_extension": {
                ext: len(files) for ext, files in categorized.items()
            },
            "files": files[:100],
            "truncated": len(files) > 100
        }
    
    def llm_query(self, prompt: str, model: str = "haiku") -> str:
        """Wrapper for LLM queries using RLM backends"""
        try:
            response = self.llm_manager.query(prompt, model)
            if response.error:
                logging.warning(f"LLM query failed: {response.error}")
                return f"[Error: {response.error}]"
            return response.content
        except Exception as e:
            logging.error(f"Exception in llm_query: {str(e)}")
            return f"[Query failed: {str(e)}]"
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM backend status"""
        return self.llm_manager.get_status()