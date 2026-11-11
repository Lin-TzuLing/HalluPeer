from __future__ import annotations
from typing import Any, Dict

def get_second_node_label(label_path: str) -> str:
    parts = [p.strip() for p in str(label_path).split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else "UNKNOWN"

def paper_evidence_text(paper_content: Any) -> str:
    """Rebuild paper text for evidence using the user's requested format.
    If paper_content is a dict:
        "\n".join([f"{key}: {value}" for (key,value) in paper_content.items()])
    """
    if isinstance(paper_content, str):
        return paper_content
    if isinstance(paper_content, dict):
        return "\n".join([f"{key}: {value}" for (key, value) in paper_content.items()])
    return str(paper_content)

def position_bucket(sent_idx: int, n_sents: int) -> str:
    if n_sents <= 0:
        return "unknown"
    t = sent_idx / max(1, n_sents)
    if t < 1/3:
        return "front"
    if t < 2/3:
        return "mid"
    return "back"
