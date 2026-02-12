"""
Parallel Agent Manager - Manages concurrent sub-agent execution for RLM
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import time
import logging

from .llm_backends import get_llm_manager, LLMResponse


@dataclass
class ChunkTask:
    """Task for processing a chunk"""
    id: int
    content: Any
    task_type: str = "extraction"
    query: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Result:
    """Result from chunk processing"""
    chunk_id: int
    content: Any
    processing_time_ms: float
    model_used: str = "haiku"
    error: Optional[str] = None


class ParallelAgentManager:
    """Manages parallel execution of RLM sub-agents"""
    
    def __init__(self, max_concurrent: int = 8, llm_query_fn: Optional[Callable] = None):
        self.max_concurrent = max_concurrent
        self.llm_manager = get_llm_manager()
        
        # Use provided function or create one from the LLM manager
        if llm_query_fn:
            self.llm_query_fn = llm_query_fn
        else:
            self.llm_query_fn = self.llm_manager.create_query_function()
        
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self._active_tasks = {}
        self._completed_results = []
        
        # Log which backend is being used
        status = self.llm_manager.get_status()
        logging.info(f"RLM Agent Manager initialized with {status['current']} backend")
    
    def process_chunks_sync(self, chunks: List[Dict], query: str) -> List[Result]:
        """Process chunks synchronously with parallel execution"""
        chunk_tasks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                task = ChunkTask(
                    id=chunk.get('id', i),
                    content=chunk.get('content', str(chunk)),
                    task_type="query",
                    query=query,
                    metadata={k: v for k, v in chunk.items() if k != 'content'}
                )
            else:
                task = ChunkTask(
                    id=i,
                    content=str(chunk),
                    task_type="query",
                    query=query,
                    metadata={}
                )
            chunk_tasks.append(task)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [
                executor.submit(self._process_single_chunk, task)
                for task in chunk_tasks
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    results.append(Result(
                        chunk_id=-1,
                        content=None,
                        processing_time_ms=0,
                        error=str(e)
                    ))
        
        results.sort(key=lambda r: r.chunk_id)
        return results
    
    async def process_chunks_async(self, chunks: List[Dict], query: str) -> List[Result]:
        """Process chunks asynchronously"""
        chunk_tasks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                task = ChunkTask(
                    id=chunk.get('id', i),
                    content=chunk.get('content', str(chunk)),
                    task_type="query",
                    query=query,
                    metadata={k: v for k, v in chunk.items() if k != 'content'}
                )
            else:
                task = ChunkTask(
                    id=i,
                    content=str(chunk),
                    task_type="query",
                    query=query,
                    metadata={}
                )
            chunk_tasks.append(task)
        
        tasks = []
        for batch in self._batch_tasks(chunk_tasks, self.max_concurrent):
            batch_tasks = [
                asyncio.create_task(self._process_chunk_async(task))
                for task in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    tasks.append(Result(
                        chunk_id=-1,
                        content=None,
                        processing_time_ms=0,
                        error=str(result)
                    ))
                else:
                    tasks.append(result)
        
        return tasks
    
    def _process_single_chunk(self, task: ChunkTask) -> Result:
        """Process a single chunk with real LLM processing"""
        start_time = time.time()
        
        try:
            model = self._select_model(task)
            prompt = self._build_prompt(task)
            
            # Use the LLM manager for more detailed response
            llm_response = self.llm_manager.query(prompt, model=model)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Handle both successful responses and errors
            if llm_response.error:
                logging.warning(f"LLM error for chunk {task.id}: {llm_response.error}")
                content = f"[Error processing chunk {task.id}: {llm_response.error}]"
            else:
                content = llm_response.content
            
            return Result(
                chunk_id=task.id,
                content=content,
                processing_time_ms=processing_time,
                model_used=llm_response.model_used
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logging.error(f"Exception processing chunk {task.id}: {str(e)}")
            
            return Result(
                chunk_id=task.id,
                content=f"[Processing failed for chunk {task.id}: {str(e)}]",
                processing_time_ms=processing_time,
                model_used=self._select_model(task),
                error=str(e)
            )
    
    async def _process_chunk_async(self, task: ChunkTask) -> Result:
        """Process chunk asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._process_single_chunk, task)
    
    def _select_model(self, task: ChunkTask) -> str:
        """Select appropriate model based on task type"""
        if task.task_type == "extraction":
            return "haiku"
        elif task.task_type == "analysis":
            return "sonnet"
        elif task.task_type == "synthesis":
            return "sonnet"
        else:
            return "haiku"
    
    def _build_prompt(self, task: ChunkTask) -> str:
        """Build optimized prompt for chunk processing"""
        content_preview = str(task.content)
        
        # Truncate content if too long, but preserve structure
        if len(content_preview) > 10000:
            content_preview = content_preview[:9500] + "\n... [content truncated]"
        
        chunk_info = f"Chunk {task.id}"
        if task.metadata:
            if 'line_range' in task.metadata:
                chunk_info += f" (lines {task.metadata['line_range']})"
            elif 'char_range' in task.metadata:
                chunk_info += f" (chars {task.metadata['char_range']})"
        
        if task.query:
            return f"""Analyze this data chunk and respond to the specific query below.

QUERY: {task.query}

DATA ({chunk_info}):
{content_preview}

INSTRUCTIONS:
- Focus only on information directly relevant to the query
- Be precise and concise
- If no relevant information is found, respond with "No relevant information in this chunk"
- Use bullet points for multiple findings
- Preserve important details like numbers, names, and specific terms

RESPONSE:"""
        else:
            # General extraction task
            return f"""Extract and summarize key information from this data chunk.

DATA ({chunk_info}):
{content_preview}

INSTRUCTIONS:
- Identify the main topics, concepts, or data points
- Preserve important details (numbers, names, dates, etc.)
- Organize findings in a clear, structured format
- If code: describe main functions, classes, or logic
- If data: summarize patterns, key values, or structure
- If text: extract main ideas and important facts

KEY FINDINGS:"""
    
    def _batch_tasks(self, tasks: List[ChunkTask], batch_size: int) -> List[List[ChunkTask]]:
        """Batch tasks for parallel processing"""
        batches = []
        for i in range(0, len(tasks), batch_size):
            batches.append(tasks[i:i+batch_size])
        return batches
    
    def aggregate_results(self, results: List[Result]) -> Dict[str, Any]:
        """Aggregate results from multiple chunks"""
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        
        total_time = sum(r.processing_time_ms for r in results)
        
        if all(isinstance(r.content, str) for r in successful):
            aggregated_content = '\n\n'.join(
                f"[Chunk {r.chunk_id}]:\n{r.content}"
                for r in successful
            )
        elif all(isinstance(r.content, dict) for r in successful):
            aggregated_content = {}
            for r in successful:
                if isinstance(r.content, dict):
                    aggregated_content.update(r.content)
        else:
            aggregated_content = [r.content for r in successful]
        
        return {
            "aggregated": aggregated_content,
            "chunks_processed": len(successful),
            "chunks_failed": len(failed),
            "total_processing_time_ms": total_time,
            "errors": [{"chunk": r.chunk_id, "error": r.error} for r in failed]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)