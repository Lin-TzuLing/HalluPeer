# Task 1 — Hallucination detection (binary)

Input: a review (`task1_review`) or a single review sentence (`task1_sent`) plus the paper as evidence. Output: hallucinated (1) or not (0).
Metrics: Accuracy, Precision, Recall, F1 and MCC.

## Zero-shot LLMs

```bash
# vLLM-served models (edit MODELS / DATASETS / CUDA_VISIBLE_DEVICES / PORT at the top)
bash evaluation/task_1/bash_script/task1_zeroshot.sh
# OpenAI API models (reads the key from openai_api_key.txt)
bash evaluation/task_1/bash_script/task1_zeroshot_gpt.sh

# single run
python -m evaluation.task_1.zeroshot \
    --input dataset/task_split/HalluPeer_NeurIPS2024/task1_review/test.jsonl \
    --source_data dataset/template_injection/HalluPeer_NeurIPS2024 \
    --out evaluation/evaluation_results/task1/zeroshot --evidence BM25
```

## Baselines (`baseline/`)

| Script | Method |
|---|---|
| `HHEM-2.1-Open.py` | Vectara HHEM-2.1-Open consistency classifier |
| `TrueNLI.py` | Off-the-shelf MNLI model (DeBERTa-v3-large) over evidence chunks |
| `seNtLI-style_retrieval.py` | 3-way NLI with sentence-level retrieval |
| `Refchecker.py` | Claim extraction (Mistral-7B-Claim-Extractor) + AlignScore checking (`utils/refchecker/`) |
| `RA-LLM.py` | Retrieval-augmented LLM judge (prompt variants in `evaluation/prompts/task1/RA-LLM_*`) |

Each has a launcher in `baseline/bash_script/` (`hhem.sh`, `truenli.sh`, `sentli_style_retrieval.sh`, `refchecker.sh`, `ra_llm.sh`); shared settings are in `baseline/config_baseline.yaml`. Decision thresholds are selected on the train split by the first call and written back to that file.
All baselines support `no_retrieval` and `BM25` evidence; `RA-LLM.py` also supports `LCS` evidence.

## Metrics

```bash
python -m evaluation.task_1.evaluate --source <test.jsonl> --pred <predictions.jsonl>
```
Writes `metrics/<pred_stem>.json` next to the predictions.
