"""
File-based chunking strategy for RLM processing
"""

from typing import List, Dict, Any
import os
from pathlib import Path


class FileBasedChunking:
    """Strategy for chunking files based on size and content"""
    
    def __init__(self, chunk_size: int = 50_000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def decompose(self, file_path: str, metadata: Dict) -> List[Dict]:
        """Decompose file into overlapping chunks"""
        chunks = []
        chunk_size = metadata.get('chunk_size', self.chunk_size)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.decompose_content(content, metadata)
    
    def decompose_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Decompose content string into chunks"""
        chunks = []
        chunk_size = metadata.get('chunk_size', self.chunk_size)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = self.chunk_size
        
        if '\n' in content:
            lines = content.split('\n')
            current_chunk = []
            current_size = 0
            current_start_line = 0
            
            for line in lines:
                line_size = len(line) + 1
                
                if current_size + line_size > chunk_size and current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    end_line = current_start_line + len(current_chunk)
                    chunks.append({
                        'id': len(chunks),
                        'content': chunk_content,
                        'size': current_size,
                        'line_range': (current_start_line, end_line)
                    })
                    
                    if self.overlap > 0:
                        overlap_ratio = self.overlap / max(chunk_size, 1)
                        keep_lines = int(round(len(current_chunk) * overlap_ratio))
                        keep_lines = min(max(keep_lines, 1), len(current_chunk))
                        current_chunk = current_chunk[-keep_lines:]
                        current_start_line = end_line - keep_lines
                        current_size = sum(len(l) + 1 for l in current_chunk)
                    else:
                        current_chunk = []
                        current_start_line = end_line
                        current_size = 0
                
                current_chunk.append(line)
                current_size += line_size
            
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                end_line = current_start_line + len(current_chunk)
                chunks.append({
                    'id': len(chunks),
                    'content': chunk_content,
                    'size': current_size,
                    'line_range': (current_start_line, end_line)
                })
        else:
            for i in range(0, len(content), chunk_size - self.overlap):
                chunk = content[i:i + chunk_size]
                chunks.append({
                    'id': len(chunks),
                    'content': chunk,
                    'size': len(chunk),
                    'char_range': (i, i + len(chunk))
                })
        
        return chunks
    
    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate results from chunks"""
        if not results:
            return ""
        
        if all(isinstance(r, str) for r in results):
            unique_parts = []
            seen = set()
            
            for result in results:
                lines = result.split('\n')
                for line in lines:
                    line_hash = hash(line.strip())
                    if line_hash not in seen and line.strip():
                        seen.add(line_hash)
                        unique_parts.append(line)
            
            return '\n'.join(unique_parts)
        elif all(isinstance(r, dict) for r in results):
            aggregated = {}
            for r in results:
                if isinstance(r, dict):
                    for key, value in r.items():
                        if key not in aggregated:
                            aggregated[key] = []
                        if isinstance(value, list):
                            aggregated[key].extend(value)
                        else:
                            aggregated[key].append(value)
            
            for key in aggregated:
                if len(aggregated[key]) == 1:
                    aggregated[key] = aggregated[key][0]
            
            return aggregated
        else:
            return results
