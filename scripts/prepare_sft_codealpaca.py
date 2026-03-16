#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare JSONL chat SFT file from HuggingFaceH4/CodeAlpaca_20K"
    )
    parser.add_argument("--output", default="data/codealpaca_sft_12000.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=args.split)
    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_path.open("w") as f:
        for row in ds:
            prompt = (row.get("prompt") or "").strip()
            completion = (row.get("completion") or "").strip()
            if not prompt or not completion:
                continue

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {output_path}")


if __name__ == "__main__":
    main()
