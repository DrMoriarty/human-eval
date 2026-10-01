# AGENTS.md

## Overview

Fork of OpenAI's HumanEval evaluation harness (Python, no monorepo). Entrypoint is the `evaluate_functional_correctness` console script defined in `setup.py`, backed by `human_eval/`.

## Commands

```bash
pip install -e .          # requires Python 3.7+
# Sanity check (should yield ~0.5 pass@1):
evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl
```

No test suite, linter, or CI config exists in this repo.

## Gotchas

- `human_eval/execution.py` deliberately runs untrusted model-generated code via `exec`. In this fork the `exec(check_program, exec_globals)` line (execution.py:50) is **already uncommented**, unlike the upstream repo where it ships commented out. Do not re-add the comment; do run evaluations only in a sandbox.
- Full dataset is gzipped at `data/HumanEval.jsonl.gz` (not JSONL directly); `human_eval/data.py` reads it transparently.
- Sample files are JSON Lines: one `{"task_id": ..., "completion": ...}` per line, completion only (no prompt).
- Evaluation is slow: full runs take ~16 min for 32k samples; timeout defaults mean hanging completions report "timed out". Low-RAM conditions can falsely fail correct programs (`malloc: can't allocate region`) — retry after freeing memory.
