# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **graduation thesis project** for an intelligent orchard pest and disease Q&A system based on V-KG RAG (Vector-Knowledge Graph RAG). The framework compares three methods:

1. **Pure LLM**: Direct LLM inference without retrieval
2. **Naive RAG**: Traditional vector-based RAG using Embedchain
3. **V-KG RAG** (light_rag): Graph-enhanced RAG with entity profiles and dual-level retrieval

The thesis document is at `thesis/毕业论文初稿：果园病虫害智能问答系统.md`.

## Development Commands

```bash
# Test framework setup
python test_framework.py

# Run methods to generate answers
python main.py run --method pure_llm --testset A          # Single method
python main.py run --method pure_llm,naive_rag --testset A,B  # Multiple methods
python main.py run --all                                   # All methods

# Evaluate results with LLM-as-a-Judge
python main.py evaluate

# Apply data corrections (specific rules for testset B)
python main.py fix

# Generate visualization charts
python main.py plot

# Run full pipeline (run → evaluate → fix → plot)
python main.py pipeline
```

## Environment Requirements

- Python 3.12+
- Required environment variables:
  - `OPENAI_API_KEY`: For answer generation (DashScope/Qwen API)
  - `DMX`: For LLM Judge evaluation (DMX API with GLM-5)

## Architecture

### Module Structure

```
config/          # Configuration management (API keys, paths, model settings)
  config.py      # Central Config class using dataclasses

methods/         # Method implementations
  base.py        # BaseMethod abstract class, TestRecord dataclass
  pure_llm.py    # Direct OpenAI API calls
  naive_rag.py   # Embedchain-based vector RAG
  light_rag.py   # HTTP client for external LightRAG service
  __init__.py    # METHOD_REGISTRY for method discovery

evaluation/      # LLM-as-a-Judge evaluation
  evaluator.py   # Evaluator class, LLM judge prompts, data fix rules
  plotter.py     # Matplotlib/seaborn visualization

data/            # Test data and knowledge base
  testset/       # A.json (20 factual questions), B.json (30 reasoning questions)
  documents/     # Knowledge base documents
  db/            # Vector database storage

output/          # Generated outputs
  results/       # Method outputs: {method}_output_{A|B}.json
  charts/        # Evaluation charts
```

### Key Design Patterns

1. **Method Registry Pattern**: Methods register in `METHOD_REGISTRY` dict in `methods/__init__.py`. New methods inherit from `BaseMethod` and implement `get_answer(question, max_chars)`.

2. **Dataclass-based Config**: `config/config.py` uses nested dataclasses (`APIConfig`, `PathConfig`, `EmbedchainConfig`, `LightRAGConfig`) with auto-legacy migration in `bootstrap_from_legacy()`.

3. **Testset Format**: JSON files with flexible key matching (supports Chinese/English keys via `QUESTION_KEYS` and `STANDARD_ANSWER_KEYS` tuples in `methods/base.py`).

4. **LLM-as-a-Judge**: Three-dimension scoring (Faithfulness, Comprehensiveness, Relevance) using GLM-5 model via OpenAI-compatible API.

### Method Answer Limits

- Testset A (factual questions): 100 characters max
- Testset B (reasoning questions): 350 characters max

### External Services

- **LightRAG**: Requires external service running at `http://127.0.0.1:9621/query` (configurable in `config/config.py`)
- **Embedchain**: Uses ChromaDB for vector storage (configured in `EmbedchainConfig`)

## Extending the Framework

To add a new method:

1. Create file in `methods/` inheriting from `BaseMethod`
2. Implement `get_answer(question, max_chars) -> str`
3. Optional: Implement `get_contexts(question) -> List[str]` for retrieval methods
4. Register in `methods/__init__.py` METHOD_REGISTRY
