from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from .schema import position_bucket, get_second_node_label

def iter_templates(review: Dict[str, Any]):
    # Yields (sent_idx, sent_dict, template_dict)
    sents = review.get("review_sentences", [])
    for i, s in enumerate(sents):
        for t in s.get("hallucination_injection_templates", []) or []:
            yield i, s, t

def choose_template_stratified(review: Dict[str, Any], rng: random.Random) -> Optional[Tuple[int, Dict[str, Any], Dict[str, Any], str]]:
    """Choose one hallucination template with ~1/3 probability from front/mid/back (if available).
    Returns (sent_idx, sent_dict, template_dict, pos_label).
    """
    sents = review.get("review_sentences", [])
    n = len(sents)
    by_pos = {"front": [], "mid": [], "back": []}
    for i, s, t in iter_templates(review):
        by_pos[position_bucket(i, n)].append((i, s, t))
    # Determine available buckets
    buckets = [b for b, items in by_pos.items() if items]
    if not buckets:
        return None
    # Target 1/3 each among available buckets by sampling bucket first
    b = rng.choice(buckets)
    i, s, t = rng.choice(by_pos[b])
    return i, s, t, b

def template_meta(t: Dict[str, Any]) -> Dict[str, Any]:
    full_label = t.get("hallucination_labels") or t.get("hallucination_label") or "UNKNOWN"
    return {
        "hallucination_instruction": t.get("hallucination_instruction", ""),
        "hallucination_labels": full_label,
        "label_depth1": get_second_node_label(full_label),
        "injection_aspect": t.get("injection_aspect", ""),
    }
