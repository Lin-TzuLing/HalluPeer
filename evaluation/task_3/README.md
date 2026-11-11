# Task 3 — Hallucination localization

Input: a hallucinated review (`task3_review`) plus the paper. Output: the hallucinated sentence; the gold label is a character span
`{"start_char_idx", "end_char_idx", "gold_span_text"}` into the review text.

```bash
# zero-shot (vLLM / OpenAI API)
bash evaluation/task_3/bash_script/task3_zeroshot.sh
bash evaluation/task_3/bash_script/task3_zeroshot_gpt.sh

# metrics
python -m evaluation.task_3.evaluate --source <test.jsonl> --pred <predictions.jsonl>
```

| File | Purpose |
|---|---|
| `zeroshot.py` | Zero-shot localization; prompt `evaluation/prompts/task3/task3.txt` |
| `evaluate.py` | Metrics from a predictions file and the ground truth |
| `config.yaml` | Default model, evidence and prompt settings |

Metrics: Token-F1 (BIO tags), Exact Match Span-F1 and Overlap Span-F1.
