from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import requests


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    return {"value": str(config)}


class _NoOpLogger:
    enabled = False

    def log(self, _: dict[str, float | int], step: int | None = None) -> None:
        return None

    def log_eval(
        self,
        *,
        input: str,
        output: str,
        expected: str = "",
        model_id: str = "",
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    def log_evals(self, results: list[dict[str, Any]]) -> None:
        return None

    def finish(self, status: str = "finished") -> None:
        return None


class _WandbExeLogger:
    enabled = True

    def __init__(self, wandb_module: Any, run: Any):
        self._wandb = wandb_module
        self._run = run

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def _post_evals_bulk(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        base_url = str(getattr(self._run, "url", "https://wandb.exe.xyz:8000")).rstrip("/")
        run_id = str(getattr(self._run, "id"))
        response = requests.post(
            f"{base_url}/api/runs/{run_id}/evals/bulk",
            json={"results": results},
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"wandb.exe eval upload failed ({response.status_code}): {response.text}"
            )

    def log_eval(
        self,
        *,
        input: str,
        output: str,
        expected: str = "",
        model_id: str = "",
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        result = {
            "input": input,
            "output": output,
            "expected": expected,
            "model_id": model_id,
            "score": score,
            "metadata": metadata or {},
        }
        self.log_evals([result])

    def log_evals(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return

        cleaned: list[dict[str, Any]] = []
        for row in results:
            cleaned.append(
                {
                    "input": str(row.get("input", "")),
                    "output": str(row.get("output", "")),
                    "expected": str(row.get("expected", "")),
                    "model_id": str(row.get("model_id", "")),
                    "score": (
                        None
                        if row.get("score") is None
                        else float(row.get("score"))
                    ),
                    "metadata": row.get("metadata") or {},
                }
            )

        # Newer clients may provide convenience methods; fall back to REST bulk endpoint.
        if hasattr(self._wandb, "log_evals"):
            self._wandb.log_evals(cleaned)
            return
        if hasattr(self._run, "log_evals"):
            self._run.log_evals(cleaned)
            return

        self._post_evals_bulk(cleaned)

    def finish(self, status: str = "finished") -> None:
        # Force-finish through run object first (more explicit), then global finish.
        try:
            self._run.finish(status=status)
        except Exception:
            pass
        self._wandb.finish(status=status)


def init_wandb_exe_logger(
    *,
    project: str | None,
    run_name: str,
    config: Any,
    url: str,
) -> _NoOpLogger | _WandbExeLogger:
    # if/else: enable only when a project is provided.
    if not project:
        print("[logging] wandb.exe disabled (no --wandb-exe-project provided).")
        return _NoOpLogger()

    try:
        import wandb_client as wandb
    except Exception:
        print(
            "[logging] wandb.exe disabled (wandb_client not installed). "
            "Install: pip install \"wandb-exe @ git+https://gitpad.exe.xyz/wandb-exe.git\""
        )
        return _NoOpLogger()

    run = wandb.init(
        project=project,
        name=run_name,
        config=_config_to_dict(config),
        url=url,
    )
    print(f"[logging] wandb.exe enabled project={project} run={run_name} url={url}")
    return _WandbExeLogger(wandb, run)


def log_metrics_jsonl(
    logger: _NoOpLogger | _WandbExeLogger,
    *,
    metrics_path: str | Path,
    stage_prefix: str,
) -> None:
    if not getattr(logger, "enabled", False):
        return

    path = Path(metrics_path)
    if not path.exists():
        print(f"[logging] metrics file missing, skip: {path}")
        return

    lines = [x for x in path.read_text().splitlines() if x.strip()]
    for i, line in enumerate(lines):
        row = json.loads(line)
        payload: dict[str, float | int] = {}
        for key, value in row.items():
            if isinstance(value, bool):
                payload[f"{stage_prefix}/{key}"] = 1 if value else 0
            elif isinstance(value, (int, float)):
                payload[f"{stage_prefix}/{key}"] = float(value)

        if payload:
            logger.log(payload, step=i)

    print(f"[logging] uploaded {len(lines)} rows from {path} to wandb.exe")
