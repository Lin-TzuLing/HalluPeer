# evaluation/prompts — Detector prompts

Templates contain `{evidence}` (paper text or BM25-retrieved chunks) and `{sentence}` / review text placeholders.

| File | Used for |
|---|---|
| `task1/task1.txt` | Task 1 binary detection (zero-shot) |
| `task2/task2.txt` | Task 2 hallucination-type classification (9 classes) |
| `task3/task3.txt` | Task 3 hallucinated-sentence localization |
| `task1/RA-LLM_CoT.txt`, `RA-LLM_Contrast.txt` | RA-LLM baseline variants: chain-of-thought (`--mode cot`) / sample contrast (`--mode contrast`) |
| `task1/RA-LLM_KR_{review,sent}.txt` | RA-LLM knowledge-retrieval variant (`--mode kr`) with few-shot examples, per granularity |
| `task1/RA-LLM_KR_fewshot_construct.py` | Builds the two KR prompts: one training example per hallucination type plus a clean one |
