# Hallucination Benchmark Data Generator

End-to-end builder that turns per-paper template-injected JSON into 5 evaluation datasets across review-level and sentence-level tasks.

---

## Pipeline

```
papers/*.json
   │
   ▼  Phase 1: profile_paper()       — n_sents + depth1 inventory per paper
   │
   ▼  Phase 2: stratified_split()    — iterative stratification + paper-cap
   │
   ▼  Phase 3: build_group_a/b()     — task instantiation + inverse-freq balancing
   │
   ▼
out/{splits.json, taskN_*/{train,dev,test}.jsonl}
```

### Phase 1 — Paper Profiling

For each paper, compute:
- `n_sents`: total sentences across all reviews
- `inventory`: `Counter` over depth-1 taxonomy classes (`depth1` = second segment of `"A > B > C"`, e.g. `Entity` in `"Scientific Hallucinations > Entity > Claim Entity Extension"`)
- `total_templates`: sum of inventory values

### Phase 2 — Stratified Paper-Level Split

Iterative stratification (Sechidis et al., 2011) over the 9-class depth1 inventory, with a hard `paper_cap` per split to enforce exact paper-count ratios (rounding drift goes to the largest split).
All reviews of a paper are assigned to the same split, so no paper, review or sentence is shared between splits.

**Algorithm sketch**:
1. Sort papers by `total_templates` desc (heavy papers first).
2. For each paper, pick *key label* = `argmax_{L in inv} (inv[L] / remaining_demand[L])`.
3. Assign to open split with largest deficit on that label; tie-break by paper-count deficit, then random.
4. Empty-inventory papers routed by paper-count deficit only.

### Phase 3 — Task Instantiation

Two output groups:

| Group | Datasets | Granularity | Pattern |
|---|---|---|---|
| **A** | `task1_review`, `task2_review`, `task3_review` | review-level | One minimal pair per review (clean ↔ hallu); only reviews with at least one injection template are used |
| **B** | `task1_sent`, `task2_sent` | sentence-level | Full expansion: one template per sentence |

**Group A** uses `_pick_review_template()` — within each split, weights candidates by inverse-frequency of (depth1, position bucket):
```
weight = 1/(1+class_counter[depth1]) + 1/(1+pos_counter[bucket])
```
where bucket ∈ {front, mid, back} (used only for balancing); each row records the integer percentage `position_label = round(sent_idx / n_sents * 100)`.

**Group B** balances by depth1 only:
```
weight = 1/(1+class_counter[depth1])
```
Applied per-sentence over its own template list, then `class_counter[chosen_depth1] += 1`.

---

## Usage

```bash
python data_gen/generate_data.py \
    --input  dataset/template_injection/HalluPeer_NeurIPS2024 \
    --out    dataset/task_split/HalluPeer_NeurIPS2024 \
    --datasets all \
    --train_ratio 0.80 --dev_ratio 0.10 --test_ratio 0.10 \
    --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Directory of paper `*.json` or a single file |
| `--out` | required | Output root directory |
| `--datasets` | `all` | Comma list or `all`, e.g. `task1_review,task1_sent` |
| `--train_ratio` | `0.80` | Train split fraction (auto-normalized) |
| `--dev_ratio` | `0.10` | Dev split fraction |
| `--test_ratio` | `0.10` | Test split fraction |
| `--seed` | `42` | RNG seed for splits and balancing |
| `--shuffle_rows / --no-shuffle_rows` | `False` | Shuffle rows in each split before writing |

---

## Output Layout

```
<out>/
├── splits.json                       # metadata & paper→split assignment
├── dataset_stats.json                # per-(dataset, split) row stats
├── task1_review/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── task2_review/{train,dev,test}.jsonl
├── task3_review/{train,dev,test}.jsonl
├── task1_sent/{train,dev,test}.jsonl
└── task2_sent/{train,dev,test}.jsonl
```

### `dataset_stats.json`

Per-(dataset, split) summary, plus a `total` aggregate across all splits per dataset.

| Key | Always | Task-conditional |
|---|---|---|
| `total_rows` | ✓ | — |
| `text_chars` | ✓ (count/sum/min/max/mean/median/stdev/p25/p75) | — |
| `label_counts` | — | binary tasks (`task1_review`, `task1_sent`); plain `{label: count}` |
| `label_distribution` | — | 9-class tasks (`task2_review`, `task2_sent`); `{class: "count (pct%)"}` |
| `span_char_lengths` | — | `task3_review` |
| `depth1_distribution_hallu` | — | binary tasks (hallu rows only); `{class: "count (pct%)"}` |
| `depth1_distribution` | — | `task3_review`; `{class: "count (pct%)"}` |
| `position_label_distribution_hallu` | — | review-level only |

Distribution fields are emitted as `{class: "count (pct%)"}` strings, e.g. `"Reasoning Error": "49 (33%)"`, sorted by descending count. `pct = round(count / total * 100)`.

### `splits.json`

| Key | Type | Description |
|---|---|---|
| `input` | str | Resolved absolute path of input |
| `ratios` | dict | Normalized `{train, dev, test}` fractions |
| `seed` | int | RNG seed |
| `datasets` | list[str] | Requested dataset names |
| `depth1_classes` | list[str] | Observed depth1 class names (sorted) |
| `split_counts` | dict | `{train, dev, test}` paper counts |
| `assignment` | dict | `paper_id → "train" \| "dev" \| "test"` |
| `profiles` | list[dict] | Per-paper Phase-1 result |

---

## Dataset Row Schemas

All rows have a unified top-level shape:
```
{ id, paper_id, text, label, metadata }
```

`id` is a per-dataset globally unique string `{dataset_name}_{n}` where `n` is a 0-based counter shared across `train → dev → test` (assigned in that order, before any row shuffle). Example: `task1_review_0`, `task1_review_1`, ..., `task1_review_5026`. Each dataset has its own counter starting at 0.

`metadata` is a dict whose contents vary by task & label. `label` semantics differ per task.

### `task1_review` — review-level binary detection

Strictly paired: each review contributes exactly one `label=0` row and one `label=1` row.

| Field | clean (`label=0`) | hallu (`label=1`) |
|---|---|---|
| `id` | `"task1_review_<n>"` | `"task1_review_<n>"` |
| `paper_id` | ✓ | ✓ |
| `text` | clean review (single-space join) | hallu review |
| `label` | `0` | `1` |
| `metadata` | `{review_idx, review_id}` | `{review_idx, review_id, sent_idx, sent_id, template_idx, template_id, label_depth1, position_label}` |

### `task2_review` — review-level 9-class classification

Hallu-only (pure 9-class classification).

| Field | Value |
|---|---|
| `id` | `"task2_review_<n>"` |
| `paper_id` | ✓ |
| `text` | hallu review |
| `label` | depth1 string, e.g. `"Reasoning Error"` |
| `metadata` | `{review_idx, review_id, sent_idx, sent_id, template_idx, template_id, position_label}` (no `label_depth1` — already in `label`) |

### `task3_review` — review-level span localization

Hallu-only.

| Field | Value |
|---|---|
| `id` | `"task3_review_<n>"` |
| `paper_id` | ✓ |
| `text` | hallu review |
| `label` | `{start_char_idx, end_char_idx, gold_span_text}` — char span of the injected sentence within `text` |
| `metadata` | `{review_idx, review_id, sent_idx, sent_id, template_idx, template_id, label_depth1, position_label}` |

### `task1_sent` — sentence-level binary detection

| Field | clean (`label=0`) | hallu (`label=1`) |
|---|---|---|
| `id` | `"task1_sent_<n>"` | `"task1_sent_<n>"` |
| `paper_id` | ✓ | ✓ |
| `text` | original sentence | hallu sentence |
| `label` | `0` | `1` |
| `metadata` | `{review_idx, review_id, sent_idx, sent_id}` | `{review_idx, review_id, sent_idx, sent_id, template_idx, template_id, label_depth1}` |

A sentence with at least one injection template yields a clean row and a hallu row with the same `sent_id`; other sentences yield a clean row only.

### `task2_sent` — sentence-level 9-class classification

Hallu-only.

| Field | Value |
|---|---|
| `id` | `"task2_sent_<n>"` |
| `paper_id` | ✓ |
| `text` | hallu sentence |
| `label` | depth1 string |
| `metadata` | `{review_idx, review_id, sent_idx, sent_id, template_idx, template_id}` (no `label_depth1`) |

---

## Metadata Field Reference

| Key | Type | Meaning |
|---|---|---|
| `review_idx` | int | 0-based index of review in `paper["reviews"]` |
| `review_id` | str | Reviewer identifier (e.g. `"Reviewer_QFs7"`) from the source data |
| `sent_idx` | int | 0-based index of sentence in `review["review_sentences"]` |
| `sent_id` | str | Sentence ID from the source data (e.g. `"NeurIPS2022_5236_Reviewer_QFs7_s0"`) |
| `template_idx` | int | 0-based index of chosen template in `sentence["hallucination_injection_templates"]` |
| `template_id` | str | Template ID from the source data (e.g. `"..._s0_t0"`) |
| `label_depth1` | str | Taxonomy depth-1 node, e.g. `"Reasoning Error"` |
| `position_label` | int | Integer percentage `round(sent_idx / n_sents * 100)`, range 0–100 (review-level only) |

`*_idx` fields give direct list access into the source JSON; `*_id` fields trace a row back to its source review, sentence and template.

---

## Internals

### Module layout

```
data_gen/
├── generate_data.py        # main script (this README documents)
└── utils/
    ├── __init__.py
    ├── io.py               # read_json, write_jsonl, list_json_files
    ├── schema.py           # get_second_node_label, position_bucket, paper_evidence_text
    ├── build_utils.py      # template_meta, iter_templates, choose_template_stratified
    └── span_utils.py       # build_review_with_offsets, diff_spans_in_hallu_sentence
```

### Key functions in `generate_data.py`

| Function | Purpose |
|---|---|
| `profile_paper(p)` | Phase 1 — per-paper sentence count + depth1 inventory |
| `stratified_split(profiles, ratios, rng)` | Phase 2 — iterative stratification with hard paper caps |
| `_pick_review_template(review, rng, class_counter, pos_counter)` | Group A — inverse-frequency weighted template selection |
| `build_group_a(...)` | Phase 3 Group A — review-level task1/2/3 row construction |
| `build_group_b(...)` | Phase 3 Group B — sentence-level task1/2 row construction (depth1-only balancing) |

### Source data format

Each input paper JSON conforms to:
```jsonc
{
  "paper_id": "NeurIPS2022_5236",
  "paper_content": { ... },                      // paper full text by section (evidence for evaluation)
  "reviews": [
    {
      "review_id": "Reviewer_QFs7",
      "review_sentences": [
        {
          "sent_id": "NeurIPS2022_5236_Reviewer_QFs7_s0",
          "review_sentence": "...",
          "hallucination_injection_templates": [
            {
              "template_id": "..._s0_t0",
              "hallucination_instruction": "...",
              "hallucination_labels": "A > B > C > D",
              "injection_aspect": "...",
              "hallucination_sentence": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

### Implementation details

- **Review text**: review sentences are joined with a single space (`build_review_with_offsets`); Task 3 character offsets refer to this joined `text`.
- **Hallucinated review**: the hallu review is the clean review with the chosen sentence replaced by its `hallucination_sentence`.
- **Paper-level split**: all reviews and sentences of a paper go to the same split.
- **Group A review selection**: reviews containing at least one injection template form the review-level datasets, so every review in `task1_review` appears as a (clean, hallu) pair.

---

## Example outputs

### task1_review hallu row

```jsonc
{
  "id": "task1_review_137",
  "paper_id": "NeurIPS2022_5236",
  "text": "Summary:\nThe authors propose ...",
  "label": 1,
  "metadata": {
    "review_idx": 0,
    "review_id": "Reviewer_QFs7",
    "sent_idx": 20,
    "sent_id": "NeurIPS2022_5236_Reviewer_QFs7_s20",
    "template_idx": 21,
    "template_id": "NeurIPS2022_5236_Reviewer_QFs7_s20_t21",
    "label_depth1": "Reasoning Error",
    "position_label": 71
  }
}
```

### task3_review row

```jsonc
{
  "id": "task3_review_42",
  "paper_id": "NeurIPS2022_5236",
  "text": "Summary:\nThe authors propose ...",
  "label": {
    "start_char_idx": 2341,
    "end_char_idx": 2665,
    "gold_span_text": "For instance, the authors mention ..."
  },
  "metadata": { "review_idx": 0, "review_id": "...", "sent_idx": 20, "sent_id": "...",
                "template_idx": 21, "template_id": "...",
                "label_depth1": "Reasoning Error", "position_label": 71 }
}
```

### task2_sent row

```jsonc
{
  "id": "task2_sent_8901",
  "paper_id": "NeurIPS2022_5236",
  "text": "The authors propose AgraSSt, ...",
  "label": "Entity",
  "metadata": {
    "review_idx": 0,
    "review_id": "Reviewer_QFs7",
    "sent_idx": 0,
    "sent_id": "NeurIPS2022_5236_Reviewer_QFs7_s0",
    "template_idx": 0,
    "template_id": "NeurIPS2022_5236_Reviewer_QFs7_s0_t0"
  }
}
```
