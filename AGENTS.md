# AGENTS.md

Repository instructions for coding agents operating in `D:\graduation_thesis`.

## 1) Project Overview

- Language/runtime: **Python 3.12**.
- Project domain: RAG evaluation pipeline for orchard pest/disease QA.
- Main entrypoint: `main.py` (argparse CLI with subcommands).
- Core packages:
  - `config/` → centralized dataclass configuration
  - `methods/` → method implementations (`pure_llm`, `naive_rag`, `light_rag`)
  - `evaluation/` → judge scoring + chart plotting
  - `data/` → testsets, knowledge document, local DB
  - `output/` → generated results, progress logs, charts

## 2) Canonical Commands (Verified from repo)

Run all commands from repo root (`D:\graduation_thesis`).

### Install dependencies

```bash
pip install -r requirements.txt
```

### Discover CLI usage

```bash
python main.py --help
python main.py run --help
python main.py evaluate --help
python main.py plot --help
python main.py pipeline --help
```

### Main execution commands

```bash
# Run one method on one testset
python main.py run --method pure_llm --testset A

# Run one method on multiple testsets
python main.py run --method pure_llm --testset A,B

# Run multiple methods
python main.py run --method naive_rag,light_rag

# Run all configured methods
python main.py run --all

# Evaluate generated outputs
python main.py evaluate

# Plot charts from evaluation records
python main.py plot

# Full pipeline: run + evaluate + plot
python main.py pipeline
```

### Running a single test-like check

This repo does **not** have pytest unit tests. The practical single-component check is:

```bash
python -c "from test_framework import test_config; test_config()"
```

Full functional script check:

```bash
python test_framework.py
```

Note: On some Windows/GBK terminals, `test_framework.py` may fail printing `✓/✗`.
Use UTF-8 terminal/session when possible.

## 3) Build / Lint / Test Reality

- Build: no dedicated build system; script-driven Python workflow.
- Lint/format/typecheck: no canonical config files found (`ruff`, `flake8`, `pylint`, `black`, `mypy`, `pyproject.toml` absent).
- Test framework: custom `test_framework.py` + CLI smoke checks.
- CI files: none detected in this repository.

Agent rule: do not invent non-existent mandatory commands. If asked to lint, state
that no project lint tool is configured and run available verification commands.

## 4) Dependencies and External Services

From `requirements.txt`:

- `openai` (LLM API client)
- `requests` (LightRAG HTTP call)
- `embedchain` (naive RAG implementation)
- `pandas` (aggregation/statistics)
- `matplotlib`, `seaborn` (charts)
- `tqdm` (progress bars)

Runtime assumptions:

- `OPENAI_API_KEY` for generation paths
- `DMX` for judge path (current config uses this env var)
- LightRAG service URL defaults to `http://127.0.0.1:9621/query`

## 5) Code Style Guidance (derived from current code)

### Imports and module structure

- Keep imports grouped: stdlib → third-party → local.
- Existing files frequently use `Path` + `sys.path.insert(...)` to support root imports.
- Prefer consistency with the touched file instead of broad refactors.

### Formatting and readability

- Follow existing PEP8-like spacing and wrapping style.
- Use f-strings (common pattern across the codebase).
- Preserve UTF-8 and Chinese comments/docstrings where already used.
- Avoid formatting-only edits unrelated to the task.

### Typing

- Keep explicit type annotations on public methods/functions.
- Match current style using `typing.List`, `Set`, `Tuple`, dataclass field types.
- Respect base class signatures from `methods/base.py`.

### Naming

- `snake_case`: variables, functions, methods
- `PascalCase`: classes
- `UPPER_SNAKE_CASE`: constants/prompts
- Method registry keys are contract strings: `pure_llm`, `naive_rag`, `light_rag`.

### Error handling

- Use explicit `try/except` with actionable error output.
- Keep resilience behavior for batch operations (continue when one item fails).
- Return explicit fallback/error objects where current code expects structured outputs.
- Do not silently swallow exceptions in new logic.

### Data and file I/O

- Use `pathlib.Path` for path composition.
- Ensure output directories exist before writing.
- Keep JSON writing with `ensure_ascii=False` for Chinese content.

## 6) Architectural Contracts to Respect

- `methods/base.py::BaseMethod` is the extension contract.
  - Required override: `get_answer(question, max_chars)`
  - Optional override: `get_contexts(question)`
- Add/modify methods via `methods/__init__.py::METHOD_REGISTRY`.
- Keep configuration centralized in `config/config.py` dataclasses.
- Evaluation flow:
  - scoring: `evaluation/evaluator.py`
  - plotting: `evaluation/plotter.py`

## 7) Scope and Safety Rules for Agents

- Prefer minimal, targeted diffs.
- Avoid touching `old_script/` unless explicitly requested.
- Do not commit generated artifacts unless asked:
  - `output/results/*`
  - `output/charts/*`
  - `output/*.csv`
  - `output/*.jsonl`
- Never hardcode API keys, tokens, or secrets.

## 8) Cursor/Copilot Rule File Status

Checked and not found:

- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

There are no additional IDE-agent instruction overlays in this repo currently.

## 9) Agent Verification Checklist

For code changes, run a practical subset relevant to touched modules:

1. `python main.py --help` (basic CLI health)
2. If methods changed: `python main.py run --method <method> --testset A`
3. If evaluator changed: `python main.py evaluate`
4. If plotter changed: `python main.py plot`
5. If config/bootstrap changed: `python -c "from test_framework import test_config; test_config()"`

If blocked by environment (API key, service down, Windows encoding), report it clearly
with exact command/output and continue with all non-blocked verification.

## 10) What to Avoid

- Do not add new frameworks/configs unless task asks for them.
- Do not replace Chinese domain terminology with generic English labels.
- Do not alter registry key names or output file naming conventions casually.
- Do not treat `old_script/` behavior as current source of truth.

This AGENTS.md is intentionally repository-specific and operational.
