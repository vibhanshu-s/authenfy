"""Monitored training supervisor. Inspect campaign.json without blocking Colab's kernel."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from training.recovery import RecoveryStore


def main(args):
    args.run.mkdir(parents=True, exist_ok=True)
    stages = [
        ("train", "training.train", ["--require-t4", *(["--resume"] if args.resume else [])]),
        ("benchmark", "training.benchmark", ["--require-t4", *(["--resume"] if args.resume else [])]),
        ("export", "training.export", []),
        ("plots", "training.plots", []),
        ("summary", "scripts/summarize_fusion.py", []),
    ]
    if args.recovery_dir:
        stages[0][2].extend(["--recovery-dir", str(args.recovery_dir)])
        stages[1][2].extend(["--recovery-dir", str(args.recovery_dir)])
    if args.benchmark_only:
        # Benchmark validates completed training, manifest and every weight hash.
        # Never resume optimizer training merely to continue a measured benchmark.
        stages = stages[1:]
    state = dict(status="running", stages=[], started_utc=datetime.now(timezone.utc).isoformat())
    status_path = args.run / "campaign.json"
    for name, module, options in stages:
        state.update(stage=name, updated_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(state, indent=2))
        command = [sys.executable, module] if module.endswith(".py") else [sys.executable, "-m", module]
        if name in {"train", "benchmark", "export"}:
            command.append(str(args.manifest))
        command.extend([str(args.run), *options])
        with (args.run / f"{name}.log").open("a") as output:
            output.write("\nCOMMAND: " + " ".join(command) + "\n")
            output.flush()
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        state["stages"].append(dict(name=name, exit_code=result.returncode))
        if result.returncode:
            state.update(status="failed", updated_utc=datetime.now(timezone.utc).isoformat())
            status_path.write_text(json.dumps(state, indent=2))
            return result.returncode
        if args.recovery_dir:
            RecoveryStore(args.recovery_dir).snapshot(args.run)
    state.update(status="complete", updated_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resume", action="store_true", help="Restore matching training and benchmark progress")
    parser.add_argument("--recovery-dir", type=Path, help="Persistent backup storage; copied only outside measured clip timings")
    parser.add_argument("--benchmark-only", action="store_true", help="Evaluate existing trained checkpoints without optimizer training")
    raise SystemExit(main(parser.parse_args()))
