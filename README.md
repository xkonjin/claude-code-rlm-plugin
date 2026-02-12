# RLM Plugin for Claude Code

Map-reduce for LLM context windows. Process files that exceed your model's context limit by chunking, parallel extraction, and semantic synthesis.

## What this actually does

When a file is too large to fit in a single LLM context window (200K tokens for Claude, ~800KB of text), you can't process it directly. RLM splits it into chunks, runs each chunk through an LLM in parallel, then synthesizes the per-chunk findings into a unified answer.

```
Input (1.2MB JSON, 7864 records)
  │
  ├─ Chunk 0 ──→ LLM ──→ "Diana Garcia is first, 169 records here"
  ├─ Chunk 1 ──→ LLM ──→ "167 records, users 169-335"
  ├─ ...        (49 chunks in parallel)
  └─ Chunk 48 ─→ LLM ──→ "John Smith is last, 112 records here"
  │
  └─ Synthesis (sonnet) ──→ "7,864 total records. First: Diana Garcia. Last: John Smith."
```

## When to use this

**Use RLM when your content exceeds the context window:**

- JSON files > 800KB
- CSV datasets > 100K rows
- Log files > 5MB
- Codebases with 50+ files to analyze simultaneously

**Don't use RLM when content fits in context.** For anything under ~800KB, direct processing is faster, cheaper, and more accurate. There is no token savings — RLM costs 6-26% _more_ tokens due to per-chunk overhead and the synthesis pass.

## Honest benchmarks

Tested 2026-02-12 against 1.2MB JSON (7864 user records), query: "How many records? First and last user name?"

| Metric           | Result                                                      |
| ---------------- | ----------------------------------------------------------- |
| Chunks           | 49 (parallel via OpenRouter)                                |
| Total time       | 6.2s                                                        |
| First user found | Diana Garcia (correct)                                      |
| Last user found  | John Smith (correct)                                        |
| Record count     | ~2% off (synthesis summed chunks instead of using metadata) |
| Synthesis model  | claude-sonnet-4-5 via OpenRouter                            |

### Token cost comparison

| Scenario              | Input tokens | Ratio | Verdict                              |
| --------------------- | ------------ | ----- | ------------------------------------ |
| 1.2MB JSON direct     | 310K         | 1.0x  | **Exceeds 200K window — impossible** |
| 1.2MB JSON via RLM    | 329K total   | 1.06x | Works, costs 6% more                 |
| 85KB codebase direct  | 21K          | 1.0x  | Fits in window — use direct          |
| 85KB codebase via RLM | 27K total    | 1.26x | Unnecessary, 26% more expensive      |

**RLM does not save tokens.** It enables processing content that wouldn't fit otherwise.

### What works well

- Chunking preserves 100% of facts (tested: 6/6 facts across 12 chunks)
- Boundary detection: first/last items correctly identified across chunks
- Parallel execution: 49 chunks in 6.2s via OpenRouter
- Graceful degradation: falls back through 6 backends, never crashes

### Known limitations

- Exact counts across chunks can be ~2% off (map-reduce inherent limitation)
- Claude CLI backend times out when run inside an active Claude Code session
- Synthesis quality depends on per-chunk extraction quality

## Setup

```bash
git clone https://github.com/Plasma-Projects/claude-code-rlm-plugin.git
cd claude-code-rlm-plugin
pip install anthropic  # or: pip install openai
```

### Authentication (pick one)

```bash
# Option 1: OpenRouter (recommended — 200+ models, single key)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Option 2: Anthropic API
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 3: OpenAI
export OPENAI_API_KEY="sk-..."

# Option 4: Local (Ollama)
ollama serve

# Option 5: Inside Claude Code (auto — uses claude CLI subprocess)
# No config needed, but slow (~5s per chunk)
```

**Backend priority:** Anthropic > OpenAI > OpenRouter > Local > Claude CLI > Fallback

## Usage

```python
from src import RLMPlugin

rlm = RLMPlugin()

# Process a large file with a query
result = rlm.process(
    file_path="/path/to/massive.json",
    query="What are the top 10 users by score?"
)

print(result["synthesis_applied"])  # True
print(result["result"]["aggregated"])  # Unified answer

# Process content string
result = rlm.process(
    content=large_string,
    query="Summarize key findings"
)

# Check backend status
print(rlm.get_llm_status())
```

### REPL mode

```python
with rlm.repl_session() as repl:
    repl.load_file("/path/to/data.csv")
    result = repl.evaluate("llm_query('Find anomalies', context)")
```

## Architecture

```
RLMPlugin
├── ContextRouter         — decides when to activate, selects chunking strategy
├── ParallelAgentManager  — parallel chunk processing + two-phase aggregation
│   ├── Phase 1: per-chunk extraction (haiku, parallel)
│   └── Phase 2: semantic synthesis (sonnet, single pass)
├── LLMManager            — 6 backends with auto-fallback
│   ├── AnthropicBackend
│   ├── OpenAIBackend
│   ├── OpenRouterBackend (tiered model selection)
│   ├── LocalLLMBackend   (Ollama)
│   ├── ClaudeCLIBackend  (subprocess)
│   └── SimpleFallbackBackend (rule-based)
└── REPLEngine            — interactive processing
```

### Chunking strategies

| Data type | Strategy                 | How it splits                     |
| --------- | ------------------------ | --------------------------------- |
| JSON/YAML | Structural decomposition | By top-level keys                 |
| CSV       | Row batching             | Groups of rows                    |
| Logs      | Time window              | By timestamp ranges               |
| Code      | File chunking            | By file with overlap              |
| Text      | Token chunking           | Fixed-size with edge preservation |

## Based on

[Recursive Language Models](https://arxiv.org/html/2512.24601v1) — programmatic examination and recursive processing of contexts that exceed LLM window limits.

## License

MIT
