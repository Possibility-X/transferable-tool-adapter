import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_REGISTRY = "experiments/registry.yaml"
DEFAULT_RUNS_DIR = "runs"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("registry.yaml must contain an experiments list")
    return experiments


def find_experiment(experiments, exp_id: str):
    matches = [exp for exp in experiments if exp.get("id") == exp_id]
    if not matches:
        raise KeyError(f"Unknown experiment id: {exp_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate experiment id: {exp_id}")
    return matches[0]


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_command(stage_spec: dict):
    script = stage_spec["script"]
    args = stage_spec.get("args", {})
    command = [sys.executable, script]
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            for item in value:
                command.extend([flag, format_value(item)])
        else:
            command.extend([flag, format_value(value)])
    return command


def command_to_text(command):
    return " ".join(
        f'"{part}"' if " " in part else part
        for part in command
    )


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def stage_output_path(stage_name: str, stage_spec: dict):
    args = stage_spec.get("args", {})
    if stage_name == "eval":
        return args.get("save")
    return args.get("summary-path")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_experiments(experiments, machine: str | None):
    rows = []
    for exp in experiments:
        if machine and str(exp.get("machine")) != machine:
            continue
        rows.append(
            {
                "id": exp.get("id"),
                "dataset": exp.get("dataset", "-"),
                "method": exp.get("method", "-"),
                "model": exp.get("model", "-"),
                "machine": exp.get("machine", "-"),
                "priority": exp.get("priority", "-"),
                "status": exp.get("status", "ready"),
            }
        )
    for row in rows:
        print(
            "{id:42} {dataset:10} {method:20} {model:16} "
            "machine={machine} priority={priority} status={status}".format(**row)
        )


def run_stage(exp: dict, stage_name: str, runs_dir: Path, skip_completed: bool):
    stage_spec = exp.get(stage_name)
    if not stage_spec:
        print(f"[skip] {exp['id']} has no {stage_name} stage")
        return 0

    output_path = stage_output_path(stage_name, stage_spec)
    if skip_completed and output_path and Path(output_path).exists():
        print(f"[done] {exp['id']} {stage_name}: {output_path} already exists")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{exp['id']}_{stage_name}_{timestamp}"
    log_path = runs_dir / "logs" / f"{run_id}.log"
    running_path = runs_dir / "running" / f"{run_id}.json"
    done_path = runs_dir / "done" / f"{run_id}.json"
    failed_path = runs_dir / "failed" / f"{run_id}.json"

    command = build_command(stage_spec)
    metadata = {
        "run_id": run_id,
        "experiment_id": exp["id"],
        "stage": stage_name,
        "dataset": exp.get("dataset"),
        "method": exp.get("method"),
        "model": exp.get("model"),
        "machine": exp.get("machine"),
        "priority": exp.get("priority"),
        "command": command,
        "command_text": command_to_text(command),
        "output_path": output_path,
        "log_path": str(log_path),
        "git_commit": get_git_commit(),
        "started_at": utc_now(),
    }
    write_json(running_path, metadata)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run] {metadata['command_text']}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    metadata["finished_at"] = utc_now()
    metadata["returncode"] = process.returncode
    running_path.unlink(missing_ok=True)
    if process.returncode == 0:
        write_json(done_path, metadata)
        print(f"[done] {exp['id']} {stage_name}; log={log_path}")
    else:
        write_json(failed_path, metadata)
        print(f"[failed] {exp['id']} {stage_name}; log={log_path}")
    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    experiments = load_registry(args.registry)

    if args.list:
        list_experiments(experiments, args.machine)
        return

    if not args.id:
        raise SystemExit("--id is required unless --list is used")

    exp = find_experiment(experiments, args.id)
    stages = ["train", "eval"] if args.stage == "all" else [args.stage]

    if exp.get("status") == "planned":
        raise SystemExit(f"{args.id} is marked planned: {exp.get('notes', '')}")

    if args.dry_run:
        for stage in stages:
            stage_spec = exp.get(stage)
            if not stage_spec:
                continue
            print(f"[dry-run] {stage}: {command_to_text(build_command(stage_spec))}")
        return

    runs_dir = Path(args.runs_dir)
    for stage in stages:
        returncode = run_stage(exp, stage, runs_dir, args.skip_completed)
        if returncode != 0:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
