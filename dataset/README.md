# dataset — Place the HalluPeer data here

This folder is empty in the repository. Download the benchmark from Hugging Face into it before running anything under `evaluation/`:

```bash
hf download Lin-TzuLing/HalluPeer --repo-type dataset --local-dir dataset
```

Expected layout after download:

```
dataset/
├── task_split/HalluPeer_{venue}/{task}/{train,dev,test}.jsonl     # Task 1/2/3 splits (injector: Qwen3-32B)
├── task_split/HalluPeer_{venue}/dataset_stats.json
├── task_split_{Llama,Mistral}/HalluPeer_{ICLR2024,NeurIPS2024}/... # cross-generator splits (Llama-3.3-70B / Mistral-Small-3.1-24B)
├── template_injection/HalluPeer_{venue}/{paper_id}.json            # paper text, reviews and injection templates
├── template_injection_{Llama,Mistral}/HalluPeer_{ICLR2024,NeurIPS2024}/{paper_id}.json
└── taxonomy/filtered_human.pkl                                     # final hallucination taxonomy tree (only needed to rebuild templates)
```

| Folder | Used by | As |
|---|---|---|
| `task_split*/` | all evaluators | `--input` / `--dataset_base` (task rows) |
| `template_injection*/` | all evaluators | `--source_data` (`{paper_id}.json` → `paper_content`, the evidence) |
| `template_injection/` | `data_gen/generate_data.py` | `--input` (to rebuild the splits) |
| `taxonomy/filtered_human.pkl` | `scripts/construct/construct_templates.sh` | `taxonomy_path` in `configs/template_construction.yaml` (not needed for evaluation) |

`filtered_human.pkl` is a Python pickle of the taxonomy tree; only load pickles from sources you trust.

Venues: `ICLR2019`–`ICLR2024`, `NeurIPS2021`–`NeurIPS2024`.
Tasks: `task1_review`, `task1_sent`, `task2_review`, `task2_sent`, `task3_review`.
Field descriptions are in the dataset card on Hugging Face.
