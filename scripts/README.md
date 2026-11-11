# scripts — Launchers for template construction and injection

Each script starts a vLLM server for the backbone LLM, waits until it is healthy, runs the pipeline over a list of venues, and shuts the server down on exit.
Scripts `cd` to the repository root themselves; logs are written next to the script.

| Script | Runs |
|---|---|
| `construct/construct_templates.sh` | `hallupeer.dataset_construction.hallucination_injection.construct_template` |
| `inject/inject_templates.sh` | `hallupeer.dataset_construction.hallucination_injection.inject_template` |

Edit the variables at the top of a script before running: `CUDA_VISIBLE_DEVICES`, `PORT`, `MODEL_NAME`, `DATASETS` (OpenReview venue folders), `MAX_PAPERS`, output paths.

```bash
bash scripts/construct/construct_templates.sh
bash scripts/inject/inject_templates.sh
```
