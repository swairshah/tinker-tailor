#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a JSONL SFT file from HuggingFaceH4/no_robots")
    parser.add_argument("--output", default="data/no_robots_sft_3000.jsonl")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_path.open("w") as f:
        for row in ds:
            messages = row.get("messages") or []
            if not messages:
                continue
            roles = {m.get("role") for m in messages if isinstance(m, dict)}
            if "user" not in roles or "assistant" not in roles:
                continue
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {output_path}")


if __name__ == "__main__":
    main()
