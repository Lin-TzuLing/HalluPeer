# configs — Pipeline configuration

YAML files loaded by `hallupeer.utils.load_configs([...])`; later files override earlier ones.
Relative paths are resolved from the repository root.

| File | Used by | Main keys |
|---|---|---|
| `agents.yaml` | all LLM steps | per agent role: `model_name`, `port`, `temperature`, `max_new_tokens`, `think_mode`, retries |
| `dataset.yaml` | `hallupeer/dataset` | `dataset_dir` (one OpenReview venue folder), batch sizes, review-filter prompt and output paths |
| `parse.yaml` | `hallupeer/parse_papers` | PDF directories per venue, GROBID output paths |
| `review_aspect_tagging.yaml` | aspect tagging | label granularity (`fine` / `coarse`), aspect mapping CSV, output paths |
| `taxonomy_generation.yaml` | taxonomy generation | prompt, predefined first-level nodes, `max_width` (8), `max_depth` (3), output pickle |
| `template_construction.yaml` | template construction | taxonomy pickle, aspect tags, prefilter / feasibility prompts, batch sizes, output paths |
| `template_injection.yaml` | injection | template results, injection and post-check prompts, output directory |

Command-line arguments of each module override the corresponding YAML values.
