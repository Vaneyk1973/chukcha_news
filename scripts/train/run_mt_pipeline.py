#!/usr/bin/env python3
"""Run the complete, resumable machine-translation pipeline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


REPORT_DIR = ROOT / "reports" / "mt" / "pipeline"
STATUS_PATH = REPORT_DIR / "status.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
DIRECTIONS = ("ru_ckt", "ckt_ru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true", help="rerun completed stages")
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_existing_log(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"{path.stem}.{timestamp}{path.suffix}"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / f"{path.stem}.{timestamp}.{suffix}{path.suffix}"
        suffix += 1
    path.replace(archive_path)
    return archive_path


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None))[1]


def load_metrics(direction: str, model_label: str) -> dict | None:
    path = ROOT / "reports" / "mt" / direction / model_label / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary() -> dict:
    directions = {}
    for direction in DIRECTIONS:
        trained = load_metrics(direction, "trained")
        baseline = load_metrics(direction, "baseline")
        result = {"trained": trained, "baseline": baseline}
        if trained and baseline:
            result["improvement"] = {
                "bleu": trained["bleu"] - baseline["bleu"],
                "chrf": trained["chrf"] - baseline["chrf"],
                "cer": baseline["cer"] - trained["cer"],
            }
        directions[direction] = result
    return {"generated_at": utc_now(), "directions": directions}


class Pipeline:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        self.args = args
        self.config = config
        self.status = {
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "state": "running",
            "current_stage": None,
            "stages": {},
        }

    def save_status(self) -> None:
        self.status["updated_at"] = utc_now()
        write_json(STATUS_PATH, self.status)

    def run(self, name: str, command: list[str], complete: bool = False) -> None:
        if complete and not self.args.force:
            print(f"[skip] {name}: output already exists", flush=True)
            self.status["stages"][name] = {"state": "skipped", "finished_at": utc_now()}
            self.save_status()
            return

        log_path = REPORT_DIR / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        archived_log = archive_existing_log(log_path)
        self.status["current_stage"] = name
        self.status["stages"][name] = {
            "state": "running",
            "started_at": utc_now(),
            "log": str(log_path.relative_to(ROOT)),
            "command": command,
        }
        if archived_log:
            self.status["stages"][name]["archived_previous_log"] = str(
                archived_log.relative_to(ROOT)
            )
        self.save_status()
        print(f"[run] {name}; log: {log_path.relative_to(ROOT)}", flush=True)

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()

        stage = self.status["stages"][name]
        stage["finished_at"] = utc_now()
        stage["return_code"] = return_code
        stage["state"] = "completed" if return_code == 0 else "failed"
        self.save_status()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def train(self, direction: str) -> None:
        output_dir = resolve_path(self.config["directions"][direction]["output_dir"])
        command = [
            self.args.python,
            "scripts/train/train_mt.py",
            "--config",
            self.args.config,
            "--direction",
            direction,
        ]
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint and not self.args.force:
            command.extend(["--resume-from-checkpoint", str(checkpoint)])
        self.run(f"train-{direction}", command, complete=(output_dir / "final").is_dir())

    def evaluate(self, direction: str, baseline: bool) -> None:
        label = "baseline" if baseline else "trained"
        command = [
            self.args.python,
            "scripts/evaluate/evaluate_mt.py",
            "--config",
            self.args.config,
            "--direction",
            direction,
        ]
        if baseline:
            command.append("--baseline")
        metrics = ROOT / "reports" / "mt" / direction / label / "metrics.json"
        self.run(f"evaluate-{label}-{direction}", command, complete=metrics.exists())

    def execute(self) -> None:
        if not self.config["training"]["enabled"]:
            raise RuntimeError("Set training.enabled=true in configs/mt.yaml before running.")

        self.save_status()
        self.run(
            "prepare-data",
            [self.args.python, "scripts/data/prepare_mt_dataset.py", "--config", self.args.config],
        )
        self.run(
            "model-smoke",
            [
                self.args.python,
                "scripts/train/validate_mt_model.py",
                "--config",
                self.args.config,
            ],
        )

        for direction in DIRECTIONS:
            self.train(direction)
            self.evaluate(direction, baseline=False)
            if not self.args.skip_baselines:
                self.evaluate(direction, baseline=True)

        write_json(SUMMARY_PATH, build_summary())
        self.status["state"] = "completed"
        self.status["current_stage"] = None
        self.status["finished_at"] = utc_now()
        self.status["summary"] = str(SUMMARY_PATH.relative_to(ROOT))
        self.save_status()
        print(f"[done] summary: {SUMMARY_PATH.relative_to(ROOT)}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pipeline = Pipeline(args, config)
    try:
        pipeline.execute()
    except BaseException as error:
        pipeline.status["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        pipeline.status["error"] = str(error) or type(error).__name__
        pipeline.status["finished_at"] = utc_now()
        pipeline.save_status()
        raise


if __name__ == "__main__":
    main()
