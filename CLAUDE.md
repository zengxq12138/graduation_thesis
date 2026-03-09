# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run methods
python main.py run --method pure_llm --testset A          # Single method, single testset
python main.py run --method naive_rag,light_rag           # Multiple methods
python main.py run --all                                   # All methods

# Evaluate and plot
python main.py evaluate                                    # Run LLM Judge scoring
python main.py plot                                        # Generate charts
python main.py pipeline                                    # Full pipeline: run + evaluate + plot

# Quick verification
python test_framework.py                                   # Functional test of all components
```

## Architecture Overview

This is a RAG evaluation framework comparing three QA approaches for orchard pest/disease domain:

```
config/          → Centralized dataclass configuration (API keys, paths, model settings)
methods/         → Method implementations inheriting from BaseMethod
                   - pure_llm: Direct LLM without retrieval
                   - naive_rag: Embedchain-based vector retrieval
                   - light_rag: External LightRAG service (requires running server)
evaluation/      → LLM Judge scoring (faithfulness, comprehensiveness, relevance) + plotting
data/            → Testsets (A.json, B.json), knowledge document, vector DB
output/          → Generated results, evaluation CSV, charts
```

### Method Extension Contract

All methods inherit from `methods/base.py::BaseMethod`:
- Required override: `get_answer(question: str, max_chars: int) -> str`
- Optional override: `get_contexts(question: str) -> List[str]`
- Register in `methods/__init__.py::METHOD_REGISTRY`

### Configuration Pattern

Configuration is centralized in `config/config.py` using nested dataclasses:
- `APIConfig`: OpenAI/ Judge API settings
- `PathConfig`: All file paths with `ensure_dirs()` auto-creation
- `EmbedchainConfig` / `LightRAGConfig`: Method-specific settings
- Access via `Config()` instance or `default_config` global

## Environment Requirements

- Python 3.12
- `OPENAI_API_KEY` - for generation (DashScope/Qwen)
- `DMX` - for judge scoring (DMX API with GLM-4.7)
- LightRAG service at `http://127.0.0.1:9621` (required for light_rag method)

## Key Code Patterns

1. **File I/O**: Use `pathlib.Path`, create dirs with `mkdir(parents=True, exist_ok=True)`, save JSON with `ensure_ascii=False` for Chinese content.

2. **Progress Resilience**: Evaluator uses JSONL append-only progress file to survive interruptions and resume from last checkpoint.

3. **Batch Processing**: Methods handle per-item errors gracefully, continuing on failure rather than aborting entire run.

4. **Chinese Domain Content**: Test data uses Chinese keys (`问题`, `标准答案`), preserve these exactly.

## Important Files

- `test_framework.py` - Functional verification of all components
- `data/testset/A.json`, `B.json` - Test questions with ground truth answers
- `data/documents/经济果林病虫害防治手册.txt` - Knowledge base document

For comprehensive operational details, see `AGENTS.md`.