PYTHON := python3
ASR_PYTHON := .venv/bin/python
MT_PYTHON := .venv/bin/python
TTS_PYTHON := .venv/bin/python
LLM_PYTHON := .venv/bin/python

.PHONY: bootstrap assets-manifest assets-verify assets-upload assets-download assets-check package-runtime-artifacts verify-runtime-artifacts showcase scan-audio prepare-parallel prepare-monolingual prepare-monolingual-trusted prepare-monolingual-with-asr prepare-mt prepare-mt-clean setup-mt download-mt-models mt-cuda-check mt-dry-run mt-clean-dry-run mt-model-smoke train-mt-ru-ckt train-mt-ckt-ru train-mt-clean-ru-ckt train-mt-clean-ckt-ru eval-mt-clean diagnose-mt-clean mt-clean translation translation-force diagnose-mt-roundtrip prepare-fieldasr setup-asr reset-asr-artifacts prepare-asr segment-asr transcribe-asr asr-smoke classify-asr-audio filter-asr-text score-asr-chukchi merge-asr-filters prepare-asr-filtered rerun-asr-filtered setup-tts prepare-tts-training-model prepare-tts-dataset prepare-tts-training-config train-tts eval-tts tts promote-asr-pseudolabels select-tts-top1h prepare-tts-dataset-top1h prepare-tts-training-config-top1h train-tts-top1h eval-tts-top1h tts-top1h setup-llm prepare-llm-chukchi-dataset train-llm-chukchi train-llm-chukchi-dry-run eval-chukchi-generation setup-serve serve serve-mock pipeline-status select-tts-pseudolabels test tree

bootstrap:
	$(PYTHON) scripts/bootstrap/download_resources.py

assets-manifest:
	$(PYTHON) scripts/bootstrap/manage_assets.py create

assets-verify:
	$(PYTHON) scripts/bootstrap/manage_assets.py verify

assets-upload:
	scripts/bootstrap/sync_bucketru.sh upload

assets-download:
	scripts/bootstrap/sync_bucketru.sh download

assets-check:
	scripts/bootstrap/sync_bucketru.sh check

package-runtime-artifacts:
	$(PYTHON) scripts/bootstrap/package_runtime_artifacts.py --clean

verify-runtime-artifacts:
	$(PYTHON) scripts/bootstrap/package_runtime_artifacts.py --verify-only

showcase:
	$(MT_PYTHON) scripts/showcase/build_showcase.py

scan-audio:
	$(PYTHON) scripts/bootstrap/scan_audio.py

prepare-parallel:
	$(PYTHON) scripts/data/prepare_parallel_corpus.py

prepare-monolingual:
	$(PYTHON) scripts/data/prepare_monolingual_corpus.py

prepare-monolingual-trusted:
	$(PYTHON) scripts/data/prepare_monolingual_corpus.py --output data/interim/chukchi_monolingual_trusted.txt

prepare-monolingual-with-asr:
	$(PYTHON) scripts/data/prepare_monolingual_corpus.py --include-asr

prepare-mt:
	$(PYTHON) scripts/data/prepare_mt_dataset.py

prepare-mt-clean:
	$(PYTHON) scripts/data/prepare_mt_dataset.py --config configs/mt_clean.yaml

.venv/.mt-ready: pyproject.toml
	$(PYTHON) -m venv .venv
	$(MT_PYTHON) -m pip install --upgrade pip
	$(MT_PYTHON) -m pip install -e '.[mt,dev]'
	touch $@

setup-mt: .venv/.mt-ready

download-mt-models: setup-mt
	$(MT_PYTHON) scripts/bootstrap/download_mt_models.py

mt-cuda-check: setup-mt
	$(MT_PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA-enabled PyTorch is required'; print(torch.cuda.get_device_name())"

mt-dry-run: prepare-mt
	$(PYTHON) scripts/train/train_mt.py --direction ru_ckt --dry-run
	$(PYTHON) scripts/train/train_mt.py --direction ckt_ru --dry-run

mt-clean-dry-run: prepare-mt-clean
	$(PYTHON) scripts/train/train_mt.py --config configs/mt_clean.yaml --direction ru_ckt --dry-run
	$(PYTHON) scripts/train/train_mt.py --config configs/mt_clean.yaml --direction ckt_ru --dry-run

mt-model-smoke: setup-mt
	$(MT_PYTHON) scripts/train/validate_mt_model.py

train-mt-ru-ckt: mt-cuda-check prepare-mt
	$(MT_PYTHON) scripts/train/train_mt.py --direction ru_ckt

train-mt-ckt-ru: mt-cuda-check prepare-mt
	$(MT_PYTHON) scripts/train/train_mt.py --direction ckt_ru

train-mt-clean-ru-ckt: mt-cuda-check prepare-mt-clean
	$(MT_PYTHON) scripts/train/train_mt.py --config configs/mt_clean.yaml --direction ru_ckt

train-mt-clean-ckt-ru: mt-cuda-check prepare-mt-clean
	$(MT_PYTHON) scripts/train/train_mt.py --config configs/mt_clean.yaml --direction ckt_ru

eval-mt-clean: setup-mt
	$(MT_PYTHON) scripts/evaluate/evaluate_mt.py --config configs/mt_clean.yaml --direction ru_ckt --normalize-chukchi --label clean
	$(MT_PYTHON) scripts/evaluate/evaluate_mt.py --config configs/mt_clean.yaml --direction ckt_ru --label clean

diagnose-mt-clean: setup-mt
	$(MT_PYTHON) scripts/evaluate/diagnose_mt_roundtrip.py --config configs/mt_clean.yaml --limit 100 --batch-size 8 --output-csv reports/mt/roundtrip_clean_100.csv --report reports/mt/roundtrip_clean_100.json

mt-clean: train-mt-clean-ru-ckt train-mt-clean-ckt-ru eval-mt-clean diagnose-mt-clean

translation: mt-cuda-check
	$(MT_PYTHON) scripts/train/run_mt_pipeline.py --python $(MT_PYTHON)

translation-force: mt-cuda-check
	$(MT_PYTHON) scripts/train/run_mt_pipeline.py --python $(MT_PYTHON) --force

diagnose-mt-roundtrip: setup-mt
	$(MT_PYTHON) scripts/evaluate/diagnose_mt_roundtrip.py --limit 100 --batch-size 8

prepare-fieldasr:
	$(PYTHON) scripts/data/prepare_fieldasr_dataset.py

.venv/.asr-ready: pyproject.toml
	$(PYTHON) -m venv .venv
	$(ASR_PYTHON) -m pip install --upgrade pip
	$(ASR_PYTHON) -m pip install -e '.[asr,dev]'
	touch $@

setup-asr: .venv/.asr-ready

.venv/.tts-ready: pyproject.toml
	$(PYTHON) -m venv .venv
	$(TTS_PYTHON) -m pip install --upgrade pip
	$(TTS_PYTHON) -m pip install -e '.[tts,asr,dev]'
	touch $@

setup-tts: .venv/.tts-ready

setup-serve: .venv/.tts-ready .venv/.mt-ready .venv/.llm-ready

.venv/.llm-ready: pyproject.toml
	$(PYTHON) -m venv .venv
	$(LLM_PYTHON) -m pip install --upgrade pip
	$(LLM_PYTHON) -m pip install -e '.[llm,dev]'
	touch $@

setup-llm: .venv/.llm-ready

reset-asr-artifacts:
	rm -rf data/interim/asr_segments
	rm -f data/interim/asr_segments.csv data/interim/asr_manifest.csv data/interim/asr_transcriptions.jsonl
	rm -rf data/interim/asr_audio_classes data/interim/asr_audio_classes_smoke
	rm -rf data/interim/asr_text_filtered data/interim/asr_chukchi_text_scores data/interim/asr_clean
	rm -f reports/asr_preprocessing.json reports/asr_audio_classification.json reports/asr_text_filtering.json reports/asr_chukchi_text_scores.json reports/asr_cleaning.json

prepare-asr: setup-asr
	$(ASR_PYTHON) scripts/data/prepare_asr.py --stage all

segment-asr: setup-asr
	$(ASR_PYTHON) scripts/data/prepare_asr.py --stage segment

transcribe-asr: setup-asr
	$(ASR_PYTHON) scripts/data/prepare_asr.py --stage transcribe

asr-smoke:
	$(PYTHON) scripts/data/prepare_asr.py --stage segment --limit 1 --dry-run --vad silence

classify-asr-audio: setup-asr
	$(ASR_PYTHON) scripts/data/classify_asr_audio.py

filter-asr-text:
	$(PYTHON) scripts/data/filter_asr_pseudolabels.py

score-asr-chukchi:
	$(PYTHON) scripts/data/prepare_monolingual_corpus.py --output data/interim/chukchi_monolingual_trusted.txt
	$(PYTHON) scripts/data/score_asr_chukchi_text.py

merge-asr-filters:
	$(PYTHON) scripts/data/merge_asr_filter_signals.py

prepare-asr-filtered:
	$(MAKE) prepare-asr
	$(MAKE) classify-asr-audio
	$(MAKE) filter-asr-text
	$(MAKE) score-asr-chukchi
	$(MAKE) merge-asr-filters

rerun-asr-filtered:
	$(MAKE) reset-asr-artifacts
	$(MAKE) prepare-asr-filtered

promote-asr-pseudolabels:
	$(PYTHON) scripts/data/promote_asr_pseudolabels.py

prepare-tts-training-model: setup-tts
	$(TTS_PYTHON) scripts/train/prepare_tts_training_model.py

prepare-tts-dataset:
	$(TTS_PYTHON) scripts/data/prepare_tts_training_dataset.py

prepare-tts-training-config:
	$(TTS_PYTHON) scripts/train/prepare_tts_training_config.py

train-tts: setup-tts prepare-tts-training-model prepare-tts-dataset prepare-tts-training-config
	$(TTS_PYTHON) scripts/train/train_tts.py

eval-tts: setup-tts
	$(TTS_PYTHON) scripts/evaluate/evaluate_tts.py

tts: promote-asr-pseudolabels select-tts-pseudolabels train-tts eval-tts

select-tts-top1h:
	$(PYTHON) scripts/data/select_top_tts_hour.py

prepare-tts-dataset-top1h:
	$(TTS_PYTHON) scripts/data/prepare_tts_training_dataset.py --config configs/tts_top1h.yaml

prepare-tts-training-config-top1h:
	$(TTS_PYTHON) scripts/train/prepare_tts_training_config.py --config configs/tts_top1h.yaml

train-tts-top1h: setup-tts prepare-tts-training-model prepare-tts-dataset-top1h prepare-tts-training-config-top1h
	$(TTS_PYTHON) scripts/train/train_tts.py --config configs/tts_top1h.yaml

eval-tts-top1h: setup-tts
	$(TTS_PYTHON) scripts/evaluate/evaluate_tts.py --config configs/tts_top1h.yaml

tts-top1h: select-tts-top1h train-tts-top1h eval-tts-top1h

prepare-llm-chukchi-dataset:
	$(PYTHON) scripts/data/prepare_llm_chukchi_dataset.py

train-llm-chukchi-dry-run: setup-llm prepare-llm-chukchi-dataset
	$(LLM_PYTHON) scripts/train/train_llm_chukchi_lora.py --dry-run

train-llm-chukchi: setup-llm prepare-llm-chukchi-dataset
	$(LLM_PYTHON) scripts/train/train_llm_chukchi_lora.py

eval-chukchi-generation: setup-mt prepare-monolingual-trusted
	$(MT_PYTHON) scripts/evaluate/evaluate_chukchi_generation.py

serve: setup-serve
	$(TTS_PYTHON) scripts/server/app.py

serve-mock: setup-serve
	$(TTS_PYTHON) scripts/server/app.py --mock-llm

pipeline-status:
	$(PYTHON) scripts/pipeline_status.py

select-tts-pseudolabels:
	$(PYTHON) scripts/data/select_tts_pseudolabels.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests

tree:
	find . -maxdepth 2 -type d | sort
