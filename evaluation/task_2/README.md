# Task 2 — Hallucination type classification

Input: a hallucinated review (`task2_review`) or sentence (`task2_sent`) plus the paper. Output: one of 9 types —
Number, Entity, Temporal, Hyperbole, Overgeneralization, Attribution Failure, Reasoning Error, Context-based meaning error, False Concatenation.

```bash
# zero-shot (vLLM / OpenAI API)
bash evaluation/task_2/bash_script/task2_zeroshot.sh
bash evaluation/task_2/bash_script/task2_zeroshot_gpt.sh

# metrics
python -m evaluation.task_2.evaluate --source <test.jsonl> --pred <predictions.jsonl>
```

| File | Purpose |
|---|---|
| `zeroshot.py` | Zero-shot classification; prompt `evaluation/prompts/task2/task2.txt` |
| `evaluate.py` | Metrics from a predictions file and the ground truth |
| `config.yaml` | Default model, evidence and prompt settings |

Metrics: Macro-F1, Micro-F1 and per-label F1.
