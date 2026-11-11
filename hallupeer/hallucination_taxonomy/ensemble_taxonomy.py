"""
Ensemble taxonomy tree builder.

python -m hallupeer.hallucination_taxonomy.ensemble_taxonomy primary.pkl other1.pkl other2.pkl \
    --threshold 0.80 --max_depth 3 --output ensemble.pkl --output_deleted deleted.json --match_all [if desired]
"""


import sys

import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hallupeer.hallucination_taxonomy.hallucination_tree import HallucinationTaxonomyTreeNode


SIMILARITY_THRESHOLD = 0.80
MODEL_NAME = 'BAAI/bge-large-en-v1.5'


def _get_node_text(node):
    return f"{node.concept}: {node.concept_desc}"


def _get_all_nodes(tree):
    nodes = []
    def _traverse(node):
        nodes.append(node)
        for child in node.children:
            _traverse(child)
    _traverse(tree)
    return nodes


def _load_tree(filepath, max_depth=None):
    tree = pickle.load(open(filepath, "rb"))
    if max_depth is not None:
        tree = tree.clone_subtree(max_depth=max_depth)
    return tree


def build_ensemble_tree(
    primary_filepath: str,
    other_filepaths: list,
    threshold: float = SIMILARITY_THRESHOLD,
    max_depth: int = None,
    output_filepath: str = None,
    output_deleted_filepath: str = None,
    model_name: str = MODEL_NAME,
    match_all: bool = False,
):
    """
    Build ensemble tree by keeping only nodes in primary tree that have
    cosine sim > threshold match in ALL other trees.

    primary_filepath        : defines the output tree structure
    other_filepaths         : compared against primary; do not affect output structure
    threshold               : cosine similarity cutoff
    max_depth               : optional depth clip applied to all trees
    output_filepath         : optional path to save filtered tree as pickle
    output_deleted_filepath : optional path to save deleted-node report as JSON
    match_all               : whether to require match in ALL other trees (default: match in ANY)
    """
    primary_tree = _load_tree(primary_filepath, max_depth)
    other_trees  = [_load_tree(fp, max_depth) for fp in other_filepaths]

    print(f"Primary tree size: {primary_tree.get_size()}")
    for i, t in enumerate(other_trees):
        print(f"Other tree [{i}] size: {t.get_size()}")

    model = SentenceTransformer(model_name)

    # Batch-encode primary nodes
    primary_nodes = _get_all_nodes(primary_tree)
    primary_texts = [_get_node_text(n) for n in primary_nodes]
    primary_embs  = model.encode(primary_texts, normalize_embeddings=True, show_progress_bar=True)

    # Batch-encode each other tree
    other_nodes_list = []
    other_embs_list  = []
    for tree in other_trees:
        nodes = _get_all_nodes(tree)
        texts = [_get_node_text(n) for n in nodes]
        emb   = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        other_nodes_list.append(nodes)
        other_embs_list.append(emb)

    # Determine which primary nodes match in ALL other trees
    sim_matrix_list = [
        cosine_similarity(primary_embs, other_emb)   # (P, Oi)
        for other_emb in other_embs_list
    ]

    matched_uuids = set()
    for i, node in enumerate(primary_nodes):
        if match_all:
            if all(sim_matrix[i].max() >= threshold for sim_matrix in sim_matrix_list):
                matched_uuids.add(node.uuid)
        else:
            if any(sim_matrix[i].max() >= threshold for sim_matrix in sim_matrix_list):
                matched_uuids.add(node.uuid)

    n_matched = len(matched_uuids)
    print(f"Matched nodes: {n_matched} / {len(primary_nodes)} (threshold={threshold})")

    # Build deleted-node report: each deleted node → top-5 similar nodes across other trees
    deleted_node_records = []
    for i, node in enumerate(primary_nodes):
        if node.uuid in matched_uuids:
            continue
        candidates = []
        for src_idx, (sim_matrix, other_nodes) in enumerate(zip(sim_matrix_list, other_nodes_list)):
            for k, (s, other_node) in enumerate(zip(sim_matrix[i], other_nodes)):
                candidates.append((float(s), src_idx, other_node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top5 = candidates[:5]
        deleted_node_records.append({
            "uuid": str(node.uuid),
            "concept": node.concept,
            "concept_desc": node.concept_desc,
            "depth": node.depth,
            "top5_similar": [
                {
                    "uuid": str(on.uuid),
                    "concept": on.concept,
                    "concept_desc": on.concept_desc,
                    "depth": on.depth,
                    "similarity": round(s, 6),
                    "source_tree_index": src_idx,
                }
                for s, src_idx, on in top5
            ],
        })

    print(f"Deleted nodes: {len(deleted_node_records)}")
    
    resolved_deleted_path = output_deleted_filepath
    if resolved_deleted_path is None and output_filepath is not None:
        base = output_filepath.rsplit(".", 1)[0] if "." in output_filepath else output_filepath
        resolved_deleted_path = base + "_deleted_nodes.json"

    if resolved_deleted_path is not None:
        with open(resolved_deleted_path, "w", encoding="utf-8") as f:
            json.dump({"deleted_nodes": deleted_node_records}, f, ensure_ascii=False, indent=2)
        print(f"Deleted-node report saved to {resolved_deleted_path}")

    # Clone primary tree, pruning unmatched nodes (subtree pruned with parent)
    def _clone_and_filter(node):
        if node.uuid not in matched_uuids:
            return None
        new_node = HallucinationTaxonomyTreeNode(
            concept=node.concept,
            concept_desc=node.concept_desc,
            depth=node.depth,
        )
        new_node._processed      = node._processed
        new_node._processed_time = node._processed_time
        new_node._is_leaf        = node._is_leaf
        for child in node.children:
            filtered_child = _clone_and_filter(child)
            if filtered_child is not None:
                new_node.add_child(filtered_child)
        return new_node

    filtered_tree = _clone_and_filter(primary_tree)

    if filtered_tree is None:
        print("Warning: root node filtered out — ensemble tree is empty.")
    else:
        print(f"Filtered tree size: {filtered_tree.get_size()}")

    if output_filepath and filtered_tree is not None:
        with open(output_filepath, 'wb') as f:
            pickle.dump(filtered_tree, f)
        print(f"Saved to {output_filepath}")

    return filtered_tree, deleted_node_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ensemble taxonomy tree")
    parser.add_argument("primary",  help="Primary tree pickle filepath")
    parser.add_argument("others",   nargs="+", help="Other tree pickle filepaths")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--max_depth", type=int,   default=None)
    parser.add_argument("--output",    type=str,   default=None, help="Output pickle filepath")
    parser.add_argument("--model",     type=str,   default=MODEL_NAME)
    parser.add_argument("--match_all", action='store_true', help="Require match in ALL other trees (default: match in ANY)")
    parser.add_argument("--output_deleted", type=str, default=None, help="Output JSON filepath for deleted-node report (default: derived from --output)")
    args = parser.parse_args()

    build_ensemble_tree(
        primary_filepath=args.primary,
        other_filepaths=args.others,
        threshold=args.threshold,
        max_depth=args.max_depth,
        output_filepath=args.output,
        output_deleted_filepath=args.output_deleted,
        model_name=args.model,
        match_all=args.match_all,
    )
