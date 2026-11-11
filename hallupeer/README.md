# hallupeer — HalluPeer dataset construction pipeline

Python package that builds HalluPeer from raw OpenReview papers and reviews.
All modules are run from the repository root as `python -m hallupeer.<module>`; settings come from `configs/*.yaml`, prompts from `prompts/`.
Every LLM call goes to an OpenAI-compatible endpoint (vLLM by default, `localhost:<port>`).

## Pipeline

| Step | Module | Input → Output |
|---|---|---|
| 1. Paper parsing | `parse_papers/parsing.py` | Paper PDFs → section-structured JSON (via GROBID) |
| 2. Review filtering + segmentation | `dataset/` (`ICLR_dataset.py`, `NeurIPS_dataset.py`, `utils/filter_review.py`) | OpenReview notes + parsed papers → filtered reviews split into sentences |
| 3. Aspect tagging | `dataset_construction/review_aspect_tagging/` | Review sentences → aspect label per sentence |
| 4. Taxonomy generation | `hallucination_taxonomy/` | LLM recursive decomposition → hallucination taxonomy tree (`.pkl`) |
| 5. Template construction | `dataset_construction/hallucination_injection/construct_template.py` | Sentences × taxonomy leaves → feasible hallucination templates |
| 6. Hallucination injection | `dataset_construction/hallucination_injection/inject_template.py` | Templates → hallucinated sentences + semantic post-check |

The per-paper output of step 6 is turned into the Task 1/2/3 benchmark splits by `data_gen/`.

## Layout

```
hallupeer/
├── agents/                 # LLM agent wrappers (one class per pipeline role)
│   ├── base.py             #   BaseAgent: async OpenAI-compatible client, retries, usage tracking
│   ├── review_filter.py    #   step 2  — keep reviews consistent with the meta-review
│   ├── review_aspect_tagger.py  # step 3
│   ├── taxonomy_generator.py    # step 4
│   ├── template_constructor.py  # step 5
│   ├── template_injector.py     # step 6
│   ├── post_checker.py          # step 6 — semantic verification of injected sentences
│   └── agents.sh           #   vLLM server launcher (`MODEL_NAME` required)
├── dataset/                # ICLR / NeurIPS loaders (paper + reviews + meta-review), review filtering
├── parse_papers/           # GROBID-based PDF parsing (config: configs/parse.yaml)
├── hallucination_taxonomy/ # taxonomy generation, ensembling and filtering — see its README
├── dataset_construction/   # aspect tagging, template construction, injection — see its README
└── utils.py                # load_configs / load_prompt_template / load_json_configs
```

Agents are created through `agents.load_agent(<role>, **cfg)`; their model name, port, temperature and retry settings live in `configs/agents.yaml`.

## Raw input data

The pipeline reads raw OpenReview data from `../datasets/openreview_data/{venue}/` (set by `dataset_dir` in `configs/dataset.yaml`, `configs/parse.yaml` and `DATASETS` in `scripts/*/*.sh`):

```
openreview_data/{venue}/
├── notes/{idx}.json              # OpenReview note dump: submission, reviews, meta-review / decision
├── paper/{idx}.pdf               # submission PDF
└── parsed_pdfs/{idx}.json        # GROBID output (python -m hallupeer.parse_papers.parsing)
```

`{idx}` is a per-venue numeric file index; the dataset `paper_id` is `{venue}_{idx}` (e.g. `NeurIPS2024_1069`).
