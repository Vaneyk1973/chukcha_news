import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train" / "run_mt_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_mt_pipeline", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_latest_checkpoint_uses_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert runner.latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_archive_existing_log_moves_previous_file(tmp_path: Path) -> None:
    log_path = tmp_path / "stage.log"
    log_path.write_text("old log", encoding="utf-8")

    archived = runner.archive_existing_log(log_path)

    assert archived is not None
    assert archived.read_text(encoding="utf-8") == "old log"
    assert archived.parent == tmp_path / "archive"
    assert not log_path.exists()


def test_build_summary_calculates_improvement(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for label, chrf, bleu, cer in (
        ("trained", 50.0, 20.0, 0.3),
        ("baseline", 40.0, 15.0, 0.5),
    ):
        path = tmp_path / "reports" / "mt" / "ru_ckt" / label / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"chrf": chrf, "bleu": bleu, "cer": cer}), encoding="utf-8"
        )

    summary = runner.build_summary()

    assert summary["directions"]["ru_ckt"]["improvement"] == {
        "bleu": 5.0,
        "chrf": 10.0,
        "cer": 0.2,
    }
