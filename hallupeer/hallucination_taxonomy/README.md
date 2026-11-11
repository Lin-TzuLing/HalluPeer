# hallucination_taxonomy — Top-down taxonomy generation

Builds the peer-review hallucination taxonomy by recursive LLM decomposition (Expand → Describe), starting from general hallucination categories (e.g. Number, Entity) down to peer-review–specific injection instructions.
Each node stores a name `h_v`, a description `δ_v` and its children `C_v`.

## Workflow

```bash
# 1. Generate one taxonomy tree per backbone LLM (max_width / max_depth in configs/taxonomy_generation.yaml)
python -m hallupeer.hallucination_taxonomy.generate_taxonomy --model_name Qwen/Qwen3-32B --port 8003 \
    --output_path results/taxonomy_generation/<tree>.pkl

# 2. Merge trees from several LLMs (embedding similarity >= threshold)
python -m hallupeer.hallucination_taxonomy.ensemble_taxonomy primary.pkl other1.pkl other2.pkl \
    --threshold 0.80 --max_depth 3 --output ensemble.pkl --output_deleted deleted.json

# 3. LLM-judged synonym detection between node pairs (prompt: prompts/hallucination_taxonomy/similar_filter.txt)
python -m hallupeer.hallucination_taxonomy.similar_filter --input_pkl ensemble.pkl --output_path similar_<model>.json

# 4. Combine synonym judgments from several LLMs into a review sheet
python hallupeer/hallucination_taxonomy/ensemble_similar_filter.py --ensemble-all a.json b.json c.json --output_csv review.csv

# 5. Apply (human-checked) keep/remove decisions to the tree
python hallupeer/hallucination_taxonomy/human_filter.py --pkl ensemble.pkl --mode csv_human --csv review_human.csv --output filtered.pkl
```

## Files

| File | Purpose |
|---|---|
| `hallucination_tree.py` | `HallucinationTaxonomyTreeNode` (name, description, children, uuid) and tree utilities |
| `generate_taxonomy.py` | Recursive expand/describe generation with resume support |
| `ensemble_taxonomy.py` | Merge multiple trees into one ensemble tree |
| `similar_filter.py` | Pairwise LLM synonym check over nodes at depth ≥ 2 |
| `ensemble_similar_filter.py` | Aggregate synonym results across LLMs (`--ensemble-all` / `--ensemble-any`), export CSV |
| `human_filter.py` | Remove nodes from the tree by UUID list or reviewed CSV |

Predefined first-level nodes are in `resources/hallucination_taxonomy/predefined_nodes.json`.
