# evaluation — Benchmark evaluation

All commands are run from the repository root.

```
evaluation/
├── task_1/                  # hallucination detection (binary): zero-shot LLMs and baselines
├── task_2/                  # hallucination-type classification (9 classes)
├── task_3/                  # hallucinated-sentence localization
├── prompts/                 # detector prompts for Tasks 1–3
└── evaluation_results/      # experiment outputs (created on first run)
```

Every task folder contains `config.yaml` (defaults), `zeroshot.py`, `bash_script/` (zero-shot sweeps), `evaluate.py` (metrics); see each folder's README.

## Inputs

| Argument | Path |
|---|---|
| `--input` / `--dataset_base` | `dataset/task_split/HalluPeer_{venue}/{task}/{split}.jsonl` |
| `--source_data` | `dataset/template_injection/HalluPeer_{venue}/` — one `{paper_id}.json` per paper; its `paper_content` field is the evidence |

Evidence modes: `no_retrieval` (full paper) or `BM25` (top-k paper chunks).

## Quick start

```bash
# Zero-shot: starts a vLLM server and sweeps models × evidence
bash evaluation/task_1/bash_script/task1_zeroshot.sh

# Single zero-shot run against an already running OpenAI-compatible server
python -m evaluation.task_1.zeroshot \
    --input dataset/task_split/HalluPeer_NeurIPS2024/task1_review/test.jsonl \
    --source_data dataset/template_injection/HalluPeer_NeurIPS2024 \
    --out evaluation/evaluation_results/task1/zeroshot --evidence BM25

# Score the predictions against the ground truth
python -m evaluation.task_1.evaluate \
    --source dataset/task_split/HalluPeer_NeurIPS2024/task1_review/test.jsonl \
    --pred evaluation/evaluation_results/task1/zeroshot/HalluPeer_NeurIPS2024/task1_review/<predictions>.jsonl
```

## Scoring predictions

Every predictor writes a JSONL file with one row per test `id` (`pred` field). The evaluators match rows to the ground truth by `id` and write `metrics/<pred_stem>.json` next to the predictions:

```bash
python -m evaluation.task_N.evaluate \
    --source dataset/task_split/HalluPeer_NeurIPS2024/task{N}_review/test.jsonl \
    --pred   evaluation/evaluation_results/task{N}/zeroshot/.../test_result_<exp>.jsonl
```

| Task | Metrics |
|---|---|
| 1 | Accuracy, Precision, Recall, F1, MCC (plus balanced accuracy, kappa, per-type breakdown) |
| 2 | Macro-/Micro-F1 and per-label F1 (plus accuracy, MCC) |
| 3 | Token-F1, Exact Span-F1, Overlap Span-F1 (plus char-F1, IoU\@0.5 F1) |

## Model checkpoints not tracked in git

### `alignscore.ckpt` (RefChecker / AlignScore baseline, ~4.9 GB)

The RefChecker baseline (`task_1/baseline/Refchecker.py`, `--checker_mode alignscore`) loads an AlignScore (RoBERTa-large) checkpoint from `alignscore.ckpt` in the repository root (`checker.alignscore.ckpt_path` in `task_1/baseline/config_baseline.yaml`).
If the file is missing, `task_1/baseline/utils/refchecker/alignscore_checker.py` downloads it automatically. To fetch it manually:

```bash
wget -O alignscore.ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
```
