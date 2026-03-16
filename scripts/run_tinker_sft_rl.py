#!/usr/bin/env python3
"""Backward-compatible wrapper for the old script name."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
cmd = [sys.executable, "-m", "llm_tuner.cli", "pipeline", *sys.argv[1:]]
subprocess.run(cmd, cwd=ROOT, check=True)
