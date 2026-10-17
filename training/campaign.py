"""Monitored training supervisor. Inspect campaign.json without blocking Colab's kernel."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    args.run.mkdir(parents=True, exist_ok=True)
    stages = [
        ("train", "training.train", ["--require-t4", *(["--resume"] if args.resume else [])]),
        ("benchmark", "training.benchmark", ["--require-t4", *(["--resume"] if args.resume else [])]),
        ("export", "training.export", []),
        ("plots", "training.plots", []),
        ("summary", "scripts/summarize_fusion.py", []),
    ]
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
    state.update(status="complete", updated_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resume", action="store_true", help="Restore matching training and benchmark progress")
    raise SystemExit(main(parser.parse_args()))
