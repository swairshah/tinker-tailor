from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import requests
from datasets import load_dataset

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


@dataclass
class MBPPEvalResult:
    split: str
    num_samples: int
    passed: int
    pass_rate: float
    avg_completion_chars: float
    eval_rows: list[dict[str, Any]]


def _extract_python_code(text: str) -> str:
    s = text.strip()
    fenced = re.findall(r"```(?:python)?\n(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()

    # Common fallback: start from the first function/class definition.
    for marker in ("def ", "class ", "import ", "from "):
        idx = s.find(marker)
        if idx >= 0:
            return s[idx:].strip()
    return s


def _sample_code(
    *,
    model_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    base_url: str | None = None,
) -> str:
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is missing from environment.")

    oai_base_url = (base_url or TINKER_OAI_BASE_URL).rstrip("/")
    response = requests.post(
        f"{oai_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_path,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Return only Python code. No markdown fences. No explanation."
                    ),
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Sampling failed ({response.status_code}): {response.text[:400]}"
        )
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content")
    return content or ""


def _run_mbpp_tests(
    *,
    candidate_code: str,
    test_imports: list[str],
    test_list: list[str],
    timeout_sec: int,
) -> tuple[bool, str]:
    program = "\n".join(
        [
            candidate_code,
            "",
            *test_imports,
            *test_list,
            "print('ALL_TESTS_PASSED')",
        ]
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"

    if proc.returncode == 0:
        return True, "passed"

    error = (proc.stderr or proc.stdout or "execution failed").strip()
    return False, error[:500]


def run_mbpp_eval(
    *,
    model_path: str,
    split: str = "test",
    num_samples: int = 64,
    seed: int = 42,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_sec: int = 5,
    base_url: str | None = None,
) -> MBPPEvalResult:
    ds = load_dataset("mbpp", "sanitized", split=split)
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(num_samples, len(ds))))

    eval_rows: list[dict[str, Any]] = []
    passed = 0
    completion_lengths: list[int] = []

    for i, row in enumerate(ds):
        prompt = row["prompt"]
        expected_code = row["code"]
        output = _sample_code(
            model_path=model_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
        )
        completion_lengths.append(len(output))

        candidate_code = _extract_python_code(output)
        ok, detail = _run_mbpp_tests(
            candidate_code=candidate_code,
            test_imports=row.get("test_imports") or [],
            test_list=row.get("test_list") or [],
            timeout_sec=timeout_sec,
        )
        score = 1.0 if ok else 0.0
        passed += int(ok)

        eval_rows.append(
            {
                "input": prompt,
                "expected": expected_code,
                "output": output,
                "model_id": model_path,
                "score": score,
                "metadata": {
                    "suite": "mbpp",
                    "split": split,
                    "task_id": row.get("task_id"),
                    "index": i,
                    "passed": ok,
                    "detail": detail,
                },
            }
        )

    n = len(eval_rows)
    return MBPPEvalResult(
        split=split,
        num_samples=n,
        passed=passed,
        pass_rate=(passed / n if n else 0.0),
        avg_completion_chars=(sum(completion_lengths) / n if n else 0.0),
        eval_rows=eval_rows,
    )
