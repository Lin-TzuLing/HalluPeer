# dataset — Place the HalluPeer data here

This folder is empty in the repository. Download the benchmark from Hugging Face into it before running anything under `evaluation/`:

```bash
hf download TzuLing/HalluPeer --repo-type dataset --local-dir dataset
```

Expected layout after download:

```
dataset/
├── task_split/HalluPeer_{venue}/{task}/{train,dev,test}.jsonl     # Task 1/2/3 splits (injector: Qwen3-32B)
├── task_split/HalluPeer_{venue}/dataset_stats.json
├── template_injection/HalluPeer_{venue}/{paper_id}.json            # paper text, reviews and injection templates
└── taxonomy/filtered_human.json                                    # final hallucination taxonomy tree (only needed to rebuild templates)
```

| Folder | Used by | As |
|---|---|---|
| `task_split*/` | all evaluators | `--input` / `--dataset_base` (task rows) |
| `template_injection*/` | all evaluators | `--source_data` (`{paper_id}.json` → `paper_content`, the evidence) |
| `template_injection/` | `data_gen/generate_data.py` | `--input` (to rebuild the splits) |
| `taxonomy/filtered_human.json` | `scripts/construct/construct_templates.sh` | `taxonomy_path` in `configs/template_construction.yaml` (not needed for evaluation) |

`filtered_human.json` is the final taxonomy as a nested JSON tree (`concept`, `concept_desc`, `depth`, `children`); `hallupeer.hallucination_taxonomy.hallucination_tree.load_taxonomy` rebuilds the tree from it.

Venues: `ICLR2019`–`ICLR2024`, `NeurIPS2021`–`NeurIPS2024`.
Tasks: `task1_review`, `task1_sent`, `task2_review`, `task2_sent`, `task3_review`.
Field descriptions are in the dataset card on Hugging Face.
