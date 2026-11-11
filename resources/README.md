# resources — Static resources for dataset construction

| Path | Description |
|---|---|
| `hallucination_taxonomy/predefined_nodes.json` | Root and first-level hallucination categories (name + description) that seed taxonomy generation. Also read by Task 2 for the class definitions |
| `review_aspect/aspects-{fine,coarse}.csv`, `review_aspect_{fine,coarse}.json` | Review aspect inventory and category ↔ aspect mappings used for aspect tagging |
| `review_aspect/synonyms.json`, `review_aspect/utils.py` | Aspect-name normalisation and helpers, adapted from Lu et al. (see below) |

## `review_filter/` — generated at run time

`configs/dataset.yaml` points `path_filter_list` to `resources/review_filter/review_filter_list_{venue}.jsonl`. The dataset loaders (`hallupeer/dataset/`) create this directory and filter reviews against the meta-review on the fly (needs a vLLM server, see `configs/agents.yaml`); decisions are cached there so an interrupted run resumes without re-querying the LLM. The cache is not tracked in git.

## `review_aspect/` — third-party aspect inventory (Lu et al., EMNLP 2025 Findings)

The aspect inventory follows *Identifying Aspects in Peer Reviews* (Lu, Kuznetsov, Gurevych; [arXiv:2504.06910](https://arxiv.org/abs/2504.06910)); the files are adapted from the authors' repository, released under Apache-2.0:
<https://github.com/UKPLab/aspects-in-reviews>

The large LLM annotation file `annotation-llm.csv` (~5 MB, contains review text from other corpora) is **not** redistributed here. It is not read by any HalluPeer code; download it only if you want the original annotations:

```bash
# from the repository root
curl -L -o resources/review_aspect/annotation-llm.csv \
    https://raw.githubusercontent.com/UKPLab/aspects-in-reviews/main/annotation-llm.csv
```

Expected layout (the file is git-ignored):

```
resources/review_aspect/
├── aspects-coarse.csv  aspects-fine.csv        # in this repo
├── review_aspect_coarse.json  review_aspect_fine.json
├── synonyms.json  utils.py
└── annotation-llm.csv                           # optional, download as above
```

Columns: `venue, paper_id, reviewer_id, review, annotation_1 … annotation_4`. Please follow the upstream repository's license and citation requirements if you use it.
