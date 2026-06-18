const form = document.querySelector("#prompt-form");
const promptInput = document.querySelector("#prompt");
const runButton = document.querySelector("#run-button");
const statusText = document.querySelector("#status");
const newsText = document.querySelector("#news-text");
const translationText = document.querySelector("#translation-text");
const backtranslationText = document.querySelector("#backtranslation-text");
const audio = document.querySelector("#audio");
const audioMeta = document.querySelector("#audio-meta");
const presetButtons = document.querySelectorAll("[data-prompt]");
const modeInputs = document.querySelectorAll('input[name="mode"]');
const newsCaption = document.querySelector("#news-caption");
const translationCaption = document.querySelector("#translation-caption");
const modelSelects = document.querySelectorAll("[data-model-select]");
let modelOptions = {};

const stageEls = {
  news: document.querySelector('[data-stage="news"]'),
  translation: document.querySelector('[data-stage="translation"]'),
  backtranslation: document.querySelector('[data-stage="backtranslation"]'),
  tts: document.querySelector('[data-stage="tts"]'),
};

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function markStage(stage, state) {
  const el = stageEls[stage];
  if (!el) return;
  el.classList.remove("running", "done", "error");
  if (state) el.classList.add(state);
}

function resetUi() {
  const mode = selectedMode();
  newsText.textContent =
    mode === "direct_chukchi" ? "Русский этап будет пропущен." : "Пока пусто.";
  translationText.textContent = "Пока пусто.";
  backtranslationText.textContent = "Пока пусто.";
  audio.removeAttribute("src");
  audio.load();
  audioMeta.textContent = "Аудио появится после синтеза.";
  updateModeCopy();
  Object.keys(stageEls).forEach((stage) => markStage(stage, ""));
}

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "translated";
}

function updateModeCopy() {
  const direct = selectedMode() === "direct_chukchi";
  newsCaption.textContent = direct
    ? "В direct mode этот этап пропускается."
    : "LLM пишет полноценный текст новости по промпту.";
  translationCaption.textContent = direct
    ? "LLM напрямую пишет экспериментальный чукотский текст."
    : "MT модель переводит готовую русскую новость.";
  const newsSelect = document.querySelector('[data-model-select="llm_news"]');
  const directSelect = document.querySelector('[data-model-select="direct_chukchi"]');
  const mtSelect = document.querySelector('[data-model-select="mt_ru_ckt"]');
  if (newsSelect) newsSelect.disabled = direct;
  if (directSelect) directSelect.disabled = !direct;
  if (mtSelect) mtSelect.disabled = direct;
}

function descriptionId(select) {
  return `${select.id}-description`;
}

function updateModelDescription(select) {
  const group = select.dataset.modelSelect;
  const option = modelOptions[group]?.choices?.find((item) => item.key === select.value);
  const description = document.querySelector(`#${descriptionId(select)}`);
  if (description) description.textContent = option?.description || "";
}

async function loadModelOptions() {
  const response = await fetch("/api/options");
  if (!response.ok) throw new Error(`Cannot load model options: HTTP ${response.status}`);
  modelOptions = await response.json();
  modelSelects.forEach((select) => {
    const group = select.dataset.modelSelect;
    const groupOptions = modelOptions[group];
    select.textContent = "";
    (groupOptions?.choices || []).forEach((choice) => {
      const option = document.createElement("option");
      option.value = choice.key;
      option.textContent = choice.label;
      select.appendChild(option);
    });
    if (groupOptions?.default) select.value = groupOptions.default;
    updateModelDescription(select);
  });
  updateModeCopy();
}

function selectedModels() {
  const values = {};
  modelSelects.forEach((select) => {
    values[select.dataset.modelSelect] = select.value;
  });
  return values;
}

function applyEvent(event) {
  if (event.stage === "error") {
    setStatus(event.error || "Ошибка pipeline", true);
    const runningStage = Object.keys(stageEls).find((stage) =>
      stageEls[stage].classList.contains("running"),
    );
    if (runningStage === "news") {
      newsText.textContent = event.error || "Ошибка генерации новости";
    }
    if (runningStage === "translation") {
      translationText.textContent = event.error || "Ошибка перевода";
    }
    if (runningStage === "backtranslation") {
      backtranslationText.textContent = event.error || "Ошибка обратного перевода";
    }
    if (runningStage === "tts") {
      audioMeta.textContent = event.error || "Ошибка озвучки";
    }
    Object.keys(stageEls).forEach((stage) => {
      if (stageEls[stage].classList.contains("running")) markStage(stage, "error");
    });
    return;
  }

  if (event.status === "running") {
    markStage(event.stage, "running");
    setStatus(event.message || "Работаем");
  }

  if (event.stage === "news" && event.status === "skipped") {
    markStage("news", "done");
    newsText.textContent = event.text || "Русский этап пропущен.";
    setStatus("Direct Chukchi mode");
  }

  if (event.stage === "news" && event.status === "partial") {
    markStage("news", "running");
    newsText.textContent = event.text || "";
    setStatus("LLM пишет русскую новость");
  }

  if (event.stage === "news" && event.status === "done") {
    markStage("news", "done");
    newsText.textContent = event.text || "";
    setStatus("Русская новость готова");
  }

  if (event.stage === "translation" && event.status === "partial") {
    markStage("translation", "running");
    translationText.textContent = event.text || "";
    setStatus("LLM пишет чукотский текст");
  }

  if (event.stage === "translation" && event.status === "done") {
    markStage("translation", "done");
    translationText.textContent = event.text || "";
    setStatus("Перевод готов");
  }

  if (event.stage === "backtranslation" && event.status === "running") {
    markStage("backtranslation", "running");
    setStatus(event.message || "Переводим обратно на русский");
  }

  if (event.stage === "backtranslation" && event.status === "done") {
    markStage("backtranslation", "done");
    backtranslationText.textContent = event.text || "";
    setStatus("Обратный перевод готов");
  }

  if (event.stage === "tts" && event.status === "done") {
    markStage("tts", "done");
    audio.src = event.audio_url;
    audio.load();
    audioMeta.textContent = `Готово: ${event.duration_sec} сек.`;
    setStatus("Озвучка готова");
  }

  if (event.stage === "complete") {
    setStatus(`Цикл завершен: ${event.run_id}`);
  }
}

async function runPipeline(prompt, mode) {
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, mode, models: selectedModels() }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      applyEvent(JSON.parse(line));
    }
  }

  if (buffer.trim()) {
    applyEvent(JSON.parse(buffer));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const mode = selectedMode();
  if (!prompt) {
    setStatus("Введите промпт", true);
    promptInput.focus();
    return;
  }

  resetUi();
  runButton.disabled = true;
  setStatus("Запускаем pipeline");
  try {
    await runPipeline(prompt, mode);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    runButton.disabled = false;
  }
});

presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    promptInput.focus();
  });
});

modeInputs.forEach((input) => {
  input.addEventListener("change", updateModeCopy);
});

modelSelects.forEach((select) => {
  select.addEventListener("change", () => updateModelDescription(select));
});

loadModelOptions().catch((error) => setStatus(error.message, true));
