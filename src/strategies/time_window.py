"""
Time window splitting strategy for logs and time-series data
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional


class TimeWindowSplitting:
    """Split logs and time-series data by time windows"""
    
    def __init__(self, window_size_lines: int = 1000):
        self.window_size_lines = window_size_lines
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}',
            r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]',
            r'\d{10,13}',
        ]
    
    def decompose(self, file_path: str, metadata: Dict) -> List[Dict]:
        """Decompose log file into time windows"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.decompose_content(content, metadata)
    
    def decompose_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Decompose log content into time-based chunks"""
        lines = content.split('\n')
        
        has_timestamps = self._detect_timestamps(lines[:100])
        
        if has_timestamps:
            return self._decompose_by_time(lines)
        else:
            return self._decompose_by_lines(lines)
    
    def _detect_timestamps(self, sample_lines: List[str]) -> bool:
        """Detect if lines contain timestamps"""
        timestamp_count = 0
        
        for line in sample_lines:
            if line.strip():
                for pattern in self.timestamp_patterns:
                    if re.search(pattern, line):
                        timestamp_count += 1
                        break
        
        return timestamp_count > len([l for l in sample_lines if l.strip()]) * 0.5
    
    def _decompose_by_time(self, lines: List[str]) -> List[Dict]:
        """Decompose logs by time windows"""
        chunks = []
        current_chunk = []
        current_timestamp = None
        chunk_start_time = None
        chunk_end_time = None
        
        for line in lines:
            timestamp = self._extract_timestamp(line)
            
            if timestamp:
                if not chunk_start_time:
                    chunk_start_time = timestamp
                
                if len(current_chunk) >= self.window_size_lines:
                    chunk_end_time = current_timestamp or timestamp
                    chunks.append({
                        'id': len(chunks),
                        'type': 'time_window',
                        'start_time': chunk_start_time,
                        'end_time': chunk_end_time,
                        'content': '\n'.join(current_chunk),
                        'line_count': len(current_chunk),
                        'size': sum(len(l) + 1 for l in current_chunk)
                    })
                    
                    current_chunk = current_chunk[-10:] if len(current_chunk) > 10 else []
                    chunk_start_time = timestamp
                
                current_timestamp = timestamp
            
            current_chunk.append(line)
        
        if current_chunk:
            chunks.append({
                'id': len(chunks),
                'type': 'time_window',
                'start_time': chunk_start_time,
                'end_time': current_timestamp,
                'content': '\n'.join(current_chunk),
                'line_count': len(current_chunk),
                'size': sum(len(l) + 1 for l in current_chunk)
            })
        
        return chunks if chunks else [{'id': 0, 'content': '\n'.join(lines), 'type': 'raw'}]
    
    def _decompose_by_lines(self, lines: List[str]) -> List[Dict]:
        """Decompose logs by line count"""
        chunks = []
        
        for i in range(0, len(lines), self.window_size_lines):
            chunk_lines = lines[i:min(i+self.window_size_lines, len(lines))]
            chunks.append({
                'id': len(chunks),
                'type': 'line_batch',
                'line_range': [i, min(i+self.window_size_lines, len(lines))],
                'content': '\n'.join(chunk_lines),
                'line_count': len(chunk_lines),
                'size': sum(len(l) + 1 for l in chunk_lines)
            })
        
        return chunks if chunks else [{'id': 0, 'content': '\n'.join(lines), 'type': 'raw'}]
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from log line"""
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(0)
        return None
    
    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate time-windowed results"""
        if not results:
            return ""
        
        if all(isinstance(r, dict) and 'events' in r for r in results):
            all_events = []
            for r in results:
                if isinstance(r['events'], list):
                    all_events.extend(r['events'])
            
            all_events.sort(key=lambda e: e.get('timestamp', ''))
            
            return {
                'total_events': len(all_events),
                'events': all_events,
                'time_range': {
                    'start': all_events[0].get('timestamp') if all_events else None,
                    'end': all_events[-1].get('timestamp') if all_events else None
                }
            }
        elif all(isinstance(r, str) for r in results):
            return '\n\n'.join(f"[Window {i}]:\n{r}" for i, r in enumerate(results))
        else:
            return results
