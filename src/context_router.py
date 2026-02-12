"""
Context Router - Intelligent routing for RLM activation and strategy selection
"""

import os
from typing import Tuple, Dict, Optional, Any, List
from dataclasses import dataclass
from pathlib import Path

from .strategies.file_chunking import FileBasedChunking
from .strategies.structural_decomp import StructuralDecomposition
from .strategies.time_window import TimeWindowSplitting


@dataclass
class ContextData:
    """Data structure for context analysis"""
    estimated_tokens: int
    files: List[str]
    total_size_bytes: int
    data_type: Optional[str] = None
    has_structure: bool = False
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ContextData':
        """Create context data from file"""
        size = os.path.getsize(file_path)
        estimated_tokens = size // 4
        
        suffix = Path(file_path).suffix[1:].lower()
        has_structure = suffix in ['json', 'xml', 'yaml', 'csv']
        
        return cls(
            estimated_tokens=estimated_tokens,
            files=[file_path],
            total_size_bytes=size,
            data_type=suffix,
            has_structure=has_structure
        )
    
    @classmethod
    def from_content(cls, content: str) -> 'ContextData':
        """Create context data from content string"""
        size = len(content)
        estimated_tokens = len(content.split())
        
        has_structure = content.strip().startswith('{') or content.strip().startswith('[')
        
        return cls(
            estimated_tokens=estimated_tokens,
            files=[],
            total_size_bytes=size,
            data_type='text',
            has_structure=has_structure
        )
    
    @classmethod
    def from_context(cls, context: Any) -> 'ContextData':
        """Create from arbitrary context object"""
        if isinstance(context, str):
            if os.path.exists(context):
                return cls.from_file(context)
            else:
                return cls.from_content(context)
        elif isinstance(context, list):
            total_size = sum(os.path.getsize(f) for f in context if os.path.exists(f))
            return cls(
                estimated_tokens=total_size // 4,
                files=context,
                total_size_bytes=total_size,
                data_type='mixed'
            )
        else:
            return cls(
                estimated_tokens=0,
                files=[],
                total_size_bytes=0
            )
    
    def has_large_structured_data(self) -> bool:
        """Check if context has large structured data"""
        return self.has_structure and self.total_size_bytes > 100_000


class ProcessingStrategy:
    """Base class for processing strategies"""
    
    def decompose(self, file_path: str, metadata: Dict) -> List[Dict]:
        """Decompose file into chunks"""
        raise NotImplementedError
    
    def decompose_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Decompose content string into chunks"""
        raise NotImplementedError
    
    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate results from chunks"""
        raise NotImplementedError


class ContextRouter:
    """Routes context to appropriate RLM strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._init_strategies()
    
    def _init_strategies(self):
        """Initialize available strategies"""
        processing = self.config.get('processing', {})
        chunk_size = processing.get('chunk_size', 50_000)
        overlap_percent = processing.get('chunk_overlap_percent', 10)
        overlap = int(chunk_size * (overlap_percent / 100)) if overlap_percent else 0
        log_window_lines = processing.get('log_window_lines', 1000)

        self.strategies = {
            "file_chunking": FileBasedChunking(chunk_size=chunk_size, overlap=overlap),
            "structural_decomp": StructuralDecomposition(max_chunk_size=chunk_size),
            "time_window": TimeWindowSplitting(window_size_lines=log_window_lines),
        }
    
    def should_activate_rlm(self, context_data: ContextData) -> Tuple[bool, Optional[str], Dict]:
        """
        Determine if RLM should be activated
        Returns: (should_activate, strategy_name, metadata)
        """
        auto_config = self.config.get('auto_trigger', {})
        
        if not auto_config.get('enabled', True):
            return False, None, {}

        if context_data.has_large_structured_data():
            return True, "structural_decomp", {
                "split_strategy": "auto",
                "reason": "large_structured_data"
            }

        token_threshold = auto_config.get('token_count', 100_000)
        if context_data.estimated_tokens > token_threshold:
            return True, "token_chunking", {
                "chunk_size": min(50_000, token_threshold // 2),
                "reason": "token_count_exceeded"
            }
        
        size_threshold = auto_config.get('file_size_kb', 50) * 1024
        if context_data.total_size_bytes > size_threshold:
            return True, "file_chunking", {
                "chunk_size": size_threshold // 2,
                "reason": "file_size_exceeded"
            }
        
        file_threshold = auto_config.get('file_count', 10)
        if len(context_data.files) > file_threshold:
            return True, "file_parallel", {
                "max_concurrent": self.config.get('processing', {}).get('max_concurrent_agents', 8),
                "reason": "file_count_exceeded"
            }
        
        return False, None, {}
    
    def select_strategy(self, data_type: str, size: int) -> ProcessingStrategy:
        """Select appropriate processing strategy based on data type"""
        strategy_map = {
            "json": self.strategies["structural_decomp"],
            "yaml": self.strategies["structural_decomp"],
            "xml": self.strategies["structural_decomp"],
            "csv": self.strategies["structural_decomp"],
            "log": self.strategies["time_window"],
            "logs": self.strategies["time_window"],
        }
        
        if data_type in strategy_map:
            return strategy_map[data_type]
        
        if size > 1_000_000:
            return self.strategies["file_chunking"]
        
        return self.strategies["file_chunking"]
    
    def get_strategy(self, strategy_name: str) -> ProcessingStrategy:
        """Get strategy by name"""
        if strategy_name == "token_chunking" or strategy_name == "file_chunking":
            return self.strategies["file_chunking"]
        elif strategy_name == "structural_decomp":
            return self.strategies["structural_decomp"]
        elif strategy_name == "time_window":
            return self.strategies["time_window"]
        else:
            return self.strategies["file_chunking"]
