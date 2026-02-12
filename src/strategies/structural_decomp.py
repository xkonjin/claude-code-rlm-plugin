"""
Structural decomposition strategy for JSON, XML, YAML, CSV data
"""

import json
import csv
from typing import List, Dict, Any
from io import StringIO
import re


class StructuralDecomposition:
    """Decompose structured data into logical chunks"""
    
    def __init__(self, max_chunk_size: int = 50_000):
        self.max_chunk_size = max_chunk_size
    
    def decompose(self, file_path: str, metadata: Dict) -> List[Dict]:
        """Decompose file based on its structure"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'json':
            return self._decompose_json(content)
        elif file_ext == 'csv':
            return self._decompose_csv(content)
        elif file_ext in ['xml', 'html']:
            return self._decompose_xml(content)
        elif file_ext in ['yaml', 'yml']:
            return self._decompose_yaml(content)
        else:
            return self.decompose_content(content, metadata)
    
    def decompose_content(self, content: str, metadata: Dict) -> List[Dict]:
        """Decompose content based on detected structure"""
        content = content.strip()
        
        if content.startswith('{') or content.startswith('['):
            return self._decompose_json(content)
        elif content.startswith('<'):
            return self._decompose_xml(content)
        elif ',' in content.split('\n')[0] if content else False:
            return self._decompose_csv(content)
        else:
            return self._decompose_text(content)
    
    def _decompose_json(self, content: str) -> List[Dict]:
        """Decompose JSON into logical chunks"""
        chunks = []
        
        try:
            data = json.loads(content)
            
            if isinstance(data, dict):
                for key, value in data.items():
                    value_str = json.dumps(value, indent=2)
                    if len(value_str) > self.max_chunk_size:
                        sub_chunks = self._decompose_large_value(value, f"dict.{key}")
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            'id': len(chunks),
                            'type': 'dict_entry',
                            'key': key,
                            'content': value,
                            'size': len(value_str)
                        })
            
            elif isinstance(data, list):
                batch_size = max(1, len(data) // 10)
                for i in range(0, len(data), batch_size):
                    batch = data[i:min(i+batch_size, len(data))]
                    batch_str = json.dumps(batch, indent=2)
                    
                    if len(batch_str) > self.max_chunk_size and len(batch) > 1:
                        for j, item in enumerate(batch):
                            chunks.append({
                                'id': len(chunks),
                                'type': 'list_item',
                                'index': i + j,
                                'content': item,
                                'size': len(json.dumps(item))
                            })
                    else:
                        chunks.append({
                            'id': len(chunks),
                            'type': 'list_batch',
                            'range': [i, min(i+batch_size, len(data))],
                            'content': batch,
                            'size': len(batch_str)
                        })
        except json.JSONDecodeError:
            return self._decompose_text(content)
        
        return chunks if chunks else [{'id': 0, 'content': content, 'type': 'raw'}]
    
    def _decompose_large_value(self, value: Any, path: str) -> List[Dict]:
        """Recursively decompose large values"""
        chunks = []
        
        if isinstance(value, dict):
            for k, v in value.items():
                v_str = json.dumps(v)
                if len(v_str) > self.max_chunk_size // 2:
                    sub_chunks = self._decompose_large_value(v, f"{path}.{k}")
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'nested_dict',
                        'path': f"{path}.{k}",
                        'content': v,
                        'size': len(v_str)
                    })
        elif isinstance(value, list):
            batch_size = max(1, len(value) // 5)
            for i in range(0, len(value), batch_size):
                batch = value[i:min(i+batch_size, len(value))]
                chunks.append({
                    'id': len(chunks),
                    'type': 'nested_list',
                    'path': f"{path}[{i}:{i+len(batch)}]",
                    'content': batch,
                    'size': len(json.dumps(batch))
                })
        else:
            chunks.append({
                'id': len(chunks),
                'type': 'value',
                'path': path,
                'content': value,
                'size': len(str(value))
            })
        
        return chunks
    
    def _decompose_csv(self, content: str) -> List[Dict]:
        """Decompose CSV into row batches"""
        chunks = []
        
        try:
            reader = csv.DictReader(StringIO(content))
            rows = list(reader)
            
            batch_size = max(100, len(rows) // 10)
            
            for i in range(0, len(rows), batch_size):
                batch = rows[i:min(i+batch_size, len(rows))]
                chunks.append({
                    'id': len(chunks),
                    'type': 'csv_batch',
                    'row_range': [i, min(i+batch_size, len(rows))],
                    'content': batch,
                    'size': len(str(batch))
                })
        except:
            return self._decompose_text(content)
        
        return chunks if chunks else [{'id': 0, 'content': content, 'type': 'raw'}]
    
    def _decompose_xml(self, content: str) -> List[Dict]:
        """Decompose XML/HTML into element chunks"""
        chunks = []
        
        tag_pattern = re.compile(r'<(\w+)[^>]*>.*?</\1>', re.DOTALL)
        matches = [match.group(0) for match in tag_pattern.finditer(content)]
        
        if matches:
            current_chunk = []
            current_size = 0
            
            for match in matches:
                match_size = len(match)
                
                if current_size + match_size > self.max_chunk_size and current_chunk:
                    chunks.append({
                        'id': len(chunks),
                        'type': 'xml_elements',
                        'content': ''.join(current_chunk),
                        'element_count': len(current_chunk),
                        'size': current_size
                    })
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(match)
                current_size += match_size
            
            if current_chunk:
                chunks.append({
                    'id': len(chunks),
                    'type': 'xml_elements',
                    'content': ''.join(current_chunk),
                    'element_count': len(current_chunk),
                    'size': current_size
                })
        else:
            return self._decompose_text(content)
        
        return chunks if chunks else [{'id': 0, 'content': content, 'type': 'raw'}]
    
    def _decompose_yaml(self, content: str) -> List[Dict]:
        """Decompose YAML into sections"""
        chunks = []
        sections = content.split('\n---\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                chunks.append({
                    'id': len(chunks),
                    'type': 'yaml_section',
                    'section_index': i,
                    'content': section,
                    'size': len(section)
                })
        
        if not chunks:
            return self._decompose_text(content)
        
        final_chunks = []
        for chunk in chunks:
            if chunk['size'] > self.max_chunk_size:
                lines = chunk['content'].split('\n')
                sub_chunks = []
                current = []
                current_size = 0
                
                for line in lines:
                    line_size = len(line) + 1
                    if current_size + line_size > self.max_chunk_size and current:
                        sub_chunks.append({
                            'id': len(final_chunks) + len(sub_chunks),
                            'type': 'yaml_subsection',
                            'parent_section': chunk['section_index'],
                            'content': '\n'.join(current),
                            'size': current_size
                        })
                        current = []
                        current_size = 0
                    current.append(line)
                    current_size += line_size
                
                if current:
                    sub_chunks.append({
                        'id': len(final_chunks) + len(sub_chunks),
                        'type': 'yaml_subsection',
                        'parent_section': chunk['section_index'],
                        'content': '\n'.join(current),
                        'size': current_size
                    })
                
                final_chunks.extend(sub_chunks)
            else:
                chunk['id'] = len(final_chunks)
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _decompose_text(self, content: str) -> List[Dict]:
        """Fallback text decomposition"""
        chunks = []
        paragraphs = content.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para) + 2
            
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunks.append({
                    'id': len(chunks),
                    'type': 'text_block',
                    'content': '\n\n'.join(current_chunk),
                    'paragraph_count': len(current_chunk),
                    'size': current_size
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append({
                'id': len(chunks),
                'type': 'text_block',
                'content': '\n\n'.join(current_chunk),
                'paragraph_count': len(current_chunk),
                'size': current_size
            })
        
        return chunks if chunks else [{'id': 0, 'content': content, 'type': 'raw'}]
    
    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate structured results"""
        if not results:
            return None
        
        if all(isinstance(r, dict) for r in results):
            if any('type' in r and r['type'] == 'dict_entry' for r in results):
                aggregated = {}
                for r in results:
                    if 'key' in r:
                        aggregated[r['key']] = r.get('content', r)
                return aggregated
            elif any('type' in r and r['type'] == 'list_batch' for r in results):
                aggregated = []
                for r in results:
                    if 'content' in r and isinstance(r['content'], list):
                        aggregated.extend(r['content'])
                return aggregated
        
        return results
