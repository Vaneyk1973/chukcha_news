#!/usr/bin/env python3
"""Build browser-openable showcase snapshots from saved pipeline runs."""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "showcase"
SOURCE_ROOTS = [ROOT / "outputs" / "server", ROOT / "outputs" / "demo"]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
    prefer_max_new_tokens,
)


@dataclass(frozen=True)
class ShowcaseCase:
    """Document the state and behavior for the `ShowcaseCase` component."""

    slug: str
    title: str
    mode: str
    source_run: str
    prompt: str
    settings: dict[str, str]


CASES = [
    ShowcaseCase(
        slug="translated-fishermen-clean",
        title="RU -> CKT: помощь рыбакам",
        mode="translated",
        source_run="4e0ffe776b07",
        prompt="В Анадыре открыли новый пункт помощи рыбакам после сильного ветра",
        settings={
            "Русская новость": "HF Qwen 2.5 7B",
            "RU -> CKT": "Clean MT guarded",
            "CKT -> RU": "Clean backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="translated-roads-clean",
        title="RU -> CKT: дороги и тундра",
        mode="translated",
        source_run="d3b20a43d947",
        prompt="Дороги размыло, оленей переехало Камазом, старики недовольны, молодежь жалуется на рыбалку",
        settings={
            "Русская новость": "HF Qwen 2.5 7B",
            "RU -> CKT": "Clean MT guarded",
            "CKT -> RU": "Clean backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="translated-school-clean",
        title="RU -> CKT: школьная экоакция",
        mode="translated",
        source_run="6db02f3cccc2",
        prompt="Школьники из Чукотки подготовили экологическую акцию на берегу залива",
        settings={
            "Русская новость": "HF Qwen 2.5 7B",
            "RU -> CKT": "Clean MT guarded",
            "CKT -> RU": "Clean backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="translated-clinic-original",
        title="RU -> CKT: медицинский кабинет",
        mode="translated",
        source_run="0e3e26914279",
        prompt="В поселке Лаврентия заработал обновленный медицинский кабинет",
        settings={
            "Русская новость": "HF Qwen 2.5 7B",
            "RU -> CKT": "Original MT trained",
            "CKT -> RU": "Original backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="direct-roads-lora",
        title="Direct Chukchi: дороги и тундра",
        mode="direct_chukchi",
        source_run="b877e24b01ef",
        prompt="Дороги размыло, оленей переехало Камазом, старики недовольны, молодежь жалуется на рыбалку",
        settings={
            "Direct Chukchi": "Qwen 2.5 7B + Chukchi LoRA",
            "CKT -> RU": "Clean backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="direct-radio-lora",
        title="Direct Chukchi: короткая радионовость",
        mode="direct_chukchi",
        source_run="44f6de5658a1",
        prompt="Короткая местная радионовость о событиях в Чукотке",
        settings={
            "Direct Chukchi": "Qwen 2.5 7B + Chukchi LoRA",
            "CKT -> RU": "Clean backtranslator",
            "Озвучка": "MMS TTS baseline",
        },
    ),
    ShowcaseCase(
        slug="direct-region-lora",
        title="Direct Chukchi: региональное сообщение",
        mode="direct_chukchi",
        source_run="82ff400abd4d",
        prompt="Короткое региональное сообщение для проверки прямой чукотской генерации",
        settings={
            "Direct Chukchi": "Qwen 2.5 7B + Chukchi LoRA",
            "CKT -> RU": "Original backtranslator",
            "Озвучка": "Finetuned TTS top 1h",
        },
    ),
    ShowcaseCase(
        slug="direct-school-lora",
        title="Direct Chukchi: школьники и берег",
        mode="direct_chukchi",
        source_run="08c64e177374",
        prompt="Школьники из Чукотки подготовили экологическую акцию на берегу залива",
        settings={
            "Direct Chukchi": "Qwen 2.5 7B + Chukchi LoRA",
            "CKT -> RU": "Original backtranslator",
            "Озвучка": "Finetuned TTS all clean ASR",
        },
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-backtranslation",
        action="store_true",
        help="Do not fill missing CKT -> RU text with the local MT model.",
    )
    return parser.parse_args()


def find_run(run_id: str) -> Path:
    """Find run for this pipeline stage."""
    for root in SOURCE_ROOTS:
        candidate = root / run_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Saved pipeline run is missing: {run_id}")


def read_text(path: Path, default: str = "") -> str:
    """Read text for this pipeline stage."""
    return path.read_text(encoding="utf-8").strip() if path.exists() else default


def audio_duration(path: Path) -> float | None:
    """Audio duration for this pipeline stage."""
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / float(audio.getframerate())
    except Exception:
        return None


def audio_data_uri(path: Path) -> str:
    """Audio data uri for this pipeline stage."""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{payload}"


def esc(value: str) -> str:
    """Esc for this pipeline stage."""
    return html.escape(value, quote=True)


def settings_html(settings: dict[str, str]) -> str:
    """Settings html for this pipeline stage."""
    return "\n".join(
        f"<div><span>{esc(key)}</span><strong>{esc(value)}</strong></div>"
        for key, value in settings.items()
    )


def stage(title: str, caption: str, text: str) -> str:
    """Stage for this pipeline stage."""
    return f"""
      <article class="stage done">
        <header>
          <div>
            <h2>{esc(title)}</h2>
            <p>{esc(caption)}</p>
          </div>
        </header>
        <pre>{esc(text)}</pre>
      </article>
    """


class BacktranslationRuntime:
    """Document the state and behavior for the `BacktranslationRuntime` component."""

    def __init__(self) -> None:
        """Implement the `__init__` protocol hook for this object."""
        self.config = load_yaml("configs/mt_clean.yaml")
        self.direction = self.config["directions"]["ckt_ru"]
        self.model_path = resolve_path("models/checkpoints/mt_clean/ckt_ru/final")
        if not self.model_path.exists():
            self.model_path = resolve_path(self.direction["output_dir"]) / "final"

    def __enter__(self) -> "BacktranslationRuntime":
        """Acquire runtime resources required by this context manager."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        prefer_max_new_tokens(self.model)
        ensure_vocabulary_tokens(
            self.tokenizer, self.model, self.config["tokenizer"]["additional_tokens"]
        )
        ensure_language_token(
            self.tokenizer,
            self.model,
            self.direction["source_language"],
            self.direction["initialize_source_language_from"],
        )
        configure_tokenizer(self.tokenizer, self.direction)
        self.model.to(self.device).eval()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release runtime resources owned by this context manager."""
        del self.model
        del self.tokenizer
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def translate(self, text: str) -> str:
        """Translate for this pipeline stage."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        generation = {
            "num_beams": 4,
            "max_new_tokens": 160,
        }
        generation.update(generation_kwargs(self.tokenizer, self.direction))
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def build_snapshot(
    case: ShowcaseCase, output: Path, backtranslator: BacktranslationRuntime | None
) -> dict:
    """Build snapshot for this pipeline stage."""
    source = find_run(case.source_run)
    case_dir = output / case.slug
    case_dir.mkdir(parents=True, exist_ok=True)

    audio_source = source / "news_ckt.wav"
    audio_target = case_dir / "news_ckt.wav"
    if not audio_source.exists():
        raise FileNotFoundError(f"Audio is missing for {case.source_run}: {audio_source}")
    shutil.copy2(audio_source, audio_target)

    news_ru = read_text(source / "news_ru.txt")
    chukchi = read_text(source / "news_ckt.txt")
    back_ru = read_text(source / "news_back_ru.txt")
    prompt = read_text(source / "prompt.txt", case.prompt) or case.prompt
    mode_label = (
        "RU -> Chukchi -> Voice" if case.mode == "translated" else "Direct Chukchi -> Voice"
    )
    duration = audio_duration(audio_target)
    duration_text = f"{duration:.1f} сек." if duration else "готово"

    if case.mode == "direct_chukchi" and not news_ru:
        news_ru = "Direct Chukchi mode: русский этап пропущен."
    if not back_ru and backtranslator is not None and chukchi:
        back_ru = backtranslator.translate(chukchi)
        (case_dir / "news_back_ru.txt").write_text(back_ru, encoding="utf-8")
    if not back_ru:
        back_ru = "Обратный перевод не был рассчитан для этого раннего запуска."

    audio_src = audio_data_uri(audio_target)

    html_text = f"""<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(case.title)} · Chukchi News Voice</title>
    <link rel="stylesheet" href="../showcase.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Chukchi News Voice</p>
          <h1>{esc(case.title)}</h1>
          <p class="lead">{esc(mode_label)}. Сохранённый прогон pipeline с текстами этапов и прослушиваемым аудио.</p>
        </div>
        <div class="run-card">
          <div><span>Run ID</span><strong>{esc(case.source_run)}</strong></div>
          <div><span>Режим</span><strong>{esc(case.mode)}</strong></div>
          <div><span>Аудио</span><strong>{esc(duration_text)}</strong></div>
        </div>
      </section>

      <section class="prompt-card">
        <h2>Промпт</h2>
        <pre>{esc(prompt)}</pre>
      </section>

      <section class="settings">
        {settings_html(case.settings)}
      </section>

      <section class="stages">
        {stage("Русская новость", "LLM пишет новость по промпту; в direct mode этап пропускается.", news_ru)}
        {stage("Чукотский текст", "Основной текст, который затем озвучивается.", chukchi)}
        {stage("Обратный перевод", "Контрольный CKT -> RU перевод для быстрой проверки смысла.", back_ru)}
        <article class="stage done">
          <header>
            <div>
              <h2>Озвучка</h2>
              <p>Синтезированная чукотская аудиоверсия.</p>
            </div>
          </header>
          <audio controls preload="metadata" src="{audio_src}"></audio>
          <p class="muted">Длительность: {esc(duration_text)}</p>
        </article>
      </section>
    </main>
  </body>
</html>
"""
    (case_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "slug": case.slug,
        "title": case.title,
        "mode": case.mode,
        "run_id": case.source_run,
        "path": str((case_dir / "index.html").relative_to(output)),
        "audio": str(audio_target.relative_to(output)),
        "backtranslation": str((case_dir / "news_back_ru.txt").relative_to(output))
        if (case_dir / "news_back_ru.txt").exists()
        else "",
    }


def write_css(output: Path) -> None:
    """Write css for this pipeline stage."""
    (output / "showcase.css").write_text(
        """
:root {
  color-scheme: light;
  --bg: #f6f7f4;
  --ink: #1d2522;
  --muted: #60706a;
  --line: #d8ded7;
  --paper: #ffffff;
  --accent: #0f766e;
  --accent-dark: #0b4f4a;
  --warm: #b45309;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); }
.shell { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 48px; }
.hero { display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 24px; align-items: end; padding-bottom: 24px; border-bottom: 1px solid var(--line); }
.eyebrow { margin: 0 0 10px; color: var(--warm); font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; }
h1, h2, p { margin: 0; }
h1 { font-size: 42px; line-height: 1.06; }
h2 { font-size: 21px; line-height: 1.2; }
.lead { margin-top: 12px; color: var(--muted); font-size: 17px; line-height: 1.45; max-width: 760px; }
.run-card, .prompt-card, .stage, .settings { background: var(--paper); border: 1px solid var(--line); border-radius: 8px; }
.run-card { display: grid; gap: 10px; padding: 16px; }
.run-card div, .settings div { display: grid; gap: 3px; }
.run-card span, .settings span, .muted, .stage p { color: var(--muted); font-size: 14px; }
.run-card strong, .settings strong { font-size: 15px; }
.prompt-card { margin-top: 20px; padding: 18px; }
.settings { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 14px; padding: 14px; }
.stages { display: grid; gap: 16px; margin-top: 20px; }
.stage { padding: 18px; }
.stage header { margin-bottom: 12px; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; line-height: 1.45; background: #f8faf8; border: 1px solid #e5ebe4; border-radius: 8px; padding: 14px; }
audio { width: 100%; margin-top: 2px; }
.muted { margin-top: 8px; }
.index-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }
.tile { display: grid; gap: 10px; padding: 18px; background: var(--paper); border: 1px solid var(--line); border-radius: 8px; text-decoration: none; color: inherit; }
.tile:hover { border-color: rgba(15, 118, 110, 0.7); }
.pill { display: inline-flex; width: fit-content; border: 1px solid rgba(15, 118, 110, 0.35); border-radius: 999px; padding: 4px 8px; color: var(--accent-dark); background: rgba(15, 118, 110, 0.1); font-size: 13px; font-weight: 750; }
@media (max-width: 760px) {
  .hero, .settings, .index-grid { grid-template-columns: 1fr; }
  h1 { font-size: 32px; }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def render_screenshots(output: Path, snapshots: list[dict]) -> None:
    """Render screenshots for this pipeline stage."""
    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if not chrome:
        print(
            "[showcase] Chrome/Chromium is not available; HTML snapshots are ready without PNG previews."
        )
        return

    for row in snapshots:
        html_path = output / row["path"]
        screenshot_path = html_path.with_name("snapshot.png")
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1440,1400",
            f"--screenshot={screenshot_path}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        row["screenshot"] = str(screenshot_path.relative_to(output))


def write_index(output: Path, snapshots: list[dict]) -> None:
    """Write index for this pipeline stage."""
    cards = []
    for row in snapshots:
        cards.append(
            f"""
        <a class="tile" href="{esc(row["path"])}">
          <span class="pill">{esc(row["mode"])}</span>
          <h2>{esc(row["title"])}</h2>
          <p>Run ID: {esc(row["run_id"])}</p>
        </a>
            """
        )
    (output / "index.html").write_text(
        f"""<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Showcase · Chukchi News Voice</title>
    <link rel="stylesheet" href="showcase.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Showcase snapshots</p>
          <h1>Chukchi News Voice</h1>
          <p class="lead">Набор сохранённых запусков pipeline: разные режимы, разные настройки, тексты этапов и прослушиваемое аудио.</p>
        </div>
        <div class="run-card">
          <div><span>Снапшотов</span><strong>{len(snapshots)}</strong></div>
          <div><span>Формат</span><strong>HTML + WAV</strong></div>
        </div>
      </section>
      <section class="index-grid">
        {"".join(cards)}
      </section>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    write_css(output)
    if args.skip_backtranslation:
        snapshots = [build_snapshot(case, output, None) for case in CASES]
    else:
        with BacktranslationRuntime() as backtranslator:
            snapshots = [build_snapshot(case, output, backtranslator) for case in CASES]
    render_screenshots(output, snapshots)
    write_index(output, snapshots)
    print(
        json.dumps(
            {"output": str(output), "snapshots": len(snapshots)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
