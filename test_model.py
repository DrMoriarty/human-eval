#!/usr/bin/env python3
"""
Test a model served by llama-server on HumanEval.

Usage:
  # server already running:
  python test_model.py --url http://localhost:8080

  # let the script start llama-server itself:
  python test_model.py --model /path/to/model.gguf [--llama-server /path/to/llama-server]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

PROMPT_TEMPLATE = """Complete the following Python function. Reply with the code only (the function body), no explanations, no markdown fences, no tests, no examples.

```python
{prompt}
```"""


def http_json(url, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def extract_code(text):
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return text


def generate_one(url, task_id, prompt, max_tokens):
    body = {
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=prompt)}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    try:
        resp = http_json(url + "/v1/chat/completions", body)
        completion = extract_code(resp["choices"][0]["message"]["content"])
    except Exception as e:
        completion = ""
        print(f"\n{task_id}: request failed: {e}", file=sys.stderr)
    return {"task_id": task_id, "completion": completion}


def wait_for_server(url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_json(url + "/health", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="base URL of a running llama-server, e.g. http://localhost:8080")
    group.add_argument("--model", help="path to model file; llama-server will be started automatically")
    ap.add_argument("--llama-server", default="llama-server", help="path to llama-server binary (default: llama-server)")
    ap.add_argument("--port", type=int, default=8080, help="port for auto-started llama-server (default: 8080)")
    ap.add_argument("--limit", type=int, default=None, help="test only the first N tasks")
    ap.add_argument("--workers", type=int, default=4, help="parallel generation requests (default: 4)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=10.0, help="per-task execution timeout in seconds")
    args = ap.parse_args()

    started_proc = None
    if args.model:
        model_name = os.path.basename(args.model)
        url = f"http://localhost:{args.port}"
        print(f"Starting llama-server for {model_name} on port {args.port}...")
        started_proc = subprocess.Popen(
            [args.llama_server, "-m", args.model, "--port", str(args.port), "-c", "4096"]
        )
        if not wait_for_server(url):
            print("llama-server did not become healthy in time", file=sys.stderr)
            started_proc.terminate()
            sys.exit(1)
    else:
        model_name = args.url
        url = args.url.rstrip("/")
        if not wait_for_server(url, timeout=5):
            print(f"Server is not reachable at {url}", file=sys.stderr)
            sys.exit(1)

    try:
        problems = read_problems()
        if args.limit:
            problems = dict(list(problems.items())[: args.limit])

        print(f"Generating completions for {len(problems)} tasks on {url} ...")
        samples = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(generate_one, url, tid, p["prompt"], args.max_tokens): tid
                for tid, p in problems.items()
            }
            for f in tqdm(as_completed(futures), total=len(futures)):
                samples.append(f.result())
        samples.sort(key=lambda s: s["task_id"])

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        samples_file = f"samples-{ts}.jsonl"
        write_jsonl(samples_file, samples)

        problems_file = f"problems-{ts}.jsonl"
        write_jsonl(problems_file, problems.values())

        print("Evaluating...")
        pass_at_k = evaluate_functional_correctness(
            samples_file, k=[1], n_workers=4, timeout=args.timeout,
            problem_file=problems_file,
        )

        report_lines = [
            f"Model: {model_name}",
            f"URL: {url}",
            f"Date: {ts}",
            f"Tasks: {len(problems)}",
            f"Timeout: {args.timeout}s",
            f"pass@1: {pass_at_k['pass@1'] * 100:.2f}%",
            f"Samples file: {samples_file}",
        ]
        report = "\n".join(report_lines)
        result_file = f"result-{ts}.txt"
        with open(result_file, "w") as fp:
            fp.write(report + "\n")

        print("\n" + report)
        print(f"\nResults saved to {result_file}")
    finally:
        if started_proc:
            print("Stopping llama-server...")
            started_proc.terminate()
            try:
                started_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                started_proc.kill()


if __name__ == "__main__":
    main()
