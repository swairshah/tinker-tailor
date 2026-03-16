from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tinker
from dotenv import load_dotenv


def init_env() -> None:
    load_dotenv()


def ensure_model_available(model_name: str, base_url: str | None = None) -> None:
    sc = tinker.ServiceClient(base_url=base_url)
    models = {m.model_name for m in sc.get_server_capabilities().supported_models}
    if model_name not in models:
        supported = "\n".join(sorted(models))
        raise RuntimeError(
            f"Model {model_name!r} is not available in this Tinker account. Supported:\n{supported}"
        )


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        data = asdict(payload)
    else:
        data = payload
    p.write_text(json.dumps(data, indent=2))
