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
    
    def __init__(
        self,
        max_concurrent: int = 8,
        llm_query_fn: Optional[Callable] = None,
        model_map: Optional[Dict[str, str]] = None
    ):
        self.max_concurrent = max_concurrent
        self.llm_manager = get_llm_manager()
        self.model_map = model_map or {}
        
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
        if task.task_type in self.model_map:
            return self.model_map[task.task_type]
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
        content_str = str(task.content)

        # Keep up to 30K chars but preserve first and last 500 chars
        # so boundary items (first/last records) are never lost
        max_content = 30_000
        if len(content_str) > max_content:
            keep_edge = 500
            middle_budget = max_content - (keep_edge * 2) - 50
            content_preview = (
                content_str[:keep_edge + middle_budget]
                + f"\n... [{len(content_str) - max_content} chars omitted] ...\n"
                + content_str[-keep_edge:]
            )
        else:
            content_preview = content_str

        chunk_info = f"Chunk {task.id}"
        if task.metadata:
            if 'source_file' in task.metadata:
                chunk_info += f" ({task.metadata['source_file']})"
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
- Be precise and concise — short bullet points, not paragraphs
- If no relevant information is found, respond with "No relevant information in this chunk"
- Report exact counts of items/records in THIS chunk
- Note the FIRST and LAST item in this chunk (important for boundary queries)
- Preserve important details: numbers, names, dates, identifiers

RESPONSE:"""
        else:
            return f"""Extract and summarize key information from this data chunk.

DATA ({chunk_info}):
{content_preview}

INSTRUCTIONS:
- Identify the main topics, concepts, or data points
- Preserve important details (numbers, names, dates, etc.)
- Note the first and last items in this chunk
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
    
    def aggregate_results(self, results: List[Result], query: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate results from multiple chunks with semantic synthesis.

        Two-phase aggregation:
        1. Collect per-chunk findings into a combined text
        2. If a query was provided AND an LLM is available, run a synthesis
           pass that merges findings into a single coherent answer
        """
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        total_time = sum(r.processing_time_ms for r in results)

        # Phase 1: mechanical aggregation
        if all(isinstance(r.content, str) for r in successful):
            raw_aggregated = '\n\n'.join(
                f"[Chunk {r.chunk_id}]:\n{r.content}"
                for r in successful
            )
        elif all(isinstance(r.content, dict) for r in successful):
            raw_aggregated = {}
            for r in successful:
                if isinstance(r.content, dict):
                    raw_aggregated.update(r.content)
        else:
            raw_aggregated = [r.content for r in successful]

        # Phase 2: semantic synthesis (only when we have a query and string results)
        synthesized = None
        synthesis_time_ms = 0
        synthesis_model = None

        if (query
            and isinstance(raw_aggregated, str)
            and len(successful) > 1
            and not self._is_fallback_only(successful)):

            synthesized, synthesis_time_ms, synthesis_model = self._synthesize(
                raw_aggregated, query, len(successful)
            )
            total_time += synthesis_time_ms

        return {
            "aggregated": synthesized or raw_aggregated,
            "raw_chunks": raw_aggregated if synthesized else None,
            "synthesis_applied": synthesized is not None,
            "synthesis_model": synthesis_model,
            "chunks_processed": len(successful),
            "chunks_failed": len(failed),
            "total_processing_time_ms": total_time,
            "errors": [{"chunk": r.chunk_id, "error": r.error} for r in failed]
        }

    def _is_fallback_only(self, results: List[Result]) -> bool:
        """Check if all results came from the rule-based fallback (no real LLM)."""
        return all(
            r.model_used and 'fallback' in str(r.model_used).lower()
            for r in results
        )

    def _synthesize(self, chunk_findings: str, query: str, num_chunks: int) -> tuple:
        """Run a synthesis pass over all chunk findings to produce a unified answer.

        Returns (synthesized_text, time_ms, model_used) or (None, 0, None) on failure.
        """
        # Cap input to synthesis prompt — keep it under 50K chars to stay within
        # context limits for the synthesis model. If findings are larger, truncate
        # each chunk section proportionally.
        max_synthesis_input = 50_000
        if len(chunk_findings) > max_synthesis_input:
            # Truncate proportionally: keep first N chars of each chunk section
            per_chunk_budget = max_synthesis_input // max(num_chunks, 1)
            sections = chunk_findings.split('\n\n[Chunk ')
            truncated = []
            for i, section in enumerate(sections):
                if i == 0:
                    truncated.append(section[:per_chunk_budget])
                else:
                    truncated.append('[Chunk ' + section[:per_chunk_budget])
            chunk_findings = '\n\n'.join(truncated)

        synthesis_prompt = f"""You are synthesizing findings from {num_chunks} data chunks that were analyzed independently. Each chunk only saw a portion of the original data, so individual chunks may have partial answers.

ORIGINAL QUERY: {query}

FINDINGS FROM ALL CHUNKS:
{chunk_findings}

INSTRUCTIONS:
- Combine all findings into a single, coherent answer to the original query
- Resolve any contradictions between chunks (later chunks may have more complete data)
- For totals/counts: prefer authoritative metadata (e.g., "total_records" field) over summing per-chunk counts, as chunk boundaries may overlap
- Do NOT say "chunk 0 found X, chunk 3 found Y" — synthesize into one unified answer
- For "first" or "last" items: the first chunk's first item is the overall first; the last chunk's last item is the overall last
- Preserve specific details: names, numbers, dates, identifiers
- Be direct and concise — no working shown, just the answer

SYNTHESIZED ANSWER:"""

        start = time.time()
        try:
            synthesis_model = self.model_map.get("synthesis", "sonnet")
            response = self.llm_manager.query(synthesis_prompt, model=synthesis_model)
            elapsed = (time.time() - start) * 1000

            if response.error:
                # Try haiku as fallback
                fallback_model = self.model_map.get("extraction", "haiku")
                response = self.llm_manager.query(synthesis_prompt, model=fallback_model)
                elapsed = (time.time() - start) * 1000

                if response.error:
                    logging.warning(f"Synthesis failed: {response.error}")
                    return (None, 0, None)

            logging.info(f"Synthesis completed in {elapsed:.0f}ms using {response.model_used}")
            return (response.content, elapsed, response.model_used)

        except Exception as e:
            logging.error(f"Synthesis exception: {e}")
            return (None, 0, None)
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
