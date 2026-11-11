from __future__ import annotations
import difflib
from typing import List, Tuple, Dict

def build_review_with_offsets(sentences: List[str], sep: str = " ") -> Tuple[str, List[Tuple[int,int]]]:
    """Return (review_text, spans_per_sentence) where each span is [start,end) char offsets."""
    spans: List[Tuple[int,int]] = []
    parts: List[str] = []
    cur = 0
    for i, s in enumerate(sentences):
        if i > 0:
            parts.append(sep)
            cur += len(sep)
        start = cur
        parts.append(s)
        cur += len(s)
        end = cur
        spans.append((start, end))
    return "".join(parts), spans

def token_char_spans(text: str) -> List[Tuple[int,int]]:
    """Whitespace tokenization spans."""
    spans: List[Tuple[int,int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (not text[j].isspace()):
            j += 1
        spans.append((i, j))
        i = j
    return spans

def diff_spans_in_hallu_sentence(orig: str, hallu: str) -> List[Tuple[int,int]]:
    """Return char spans in hallu sentence that correspond to inserted/replaced tokens."""
    o_toks = orig.split()
    h_toks = hallu.split()
    sm = difflib.SequenceMatcher(a=o_toks, b=h_toks)
    # Map tokens to char spans in hallu
    tok_spans = token_char_spans(hallu)
    spans: List[Tuple[int,int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            if j1 < len(tok_spans):
                start = tok_spans[j1][0]
                end = tok_spans[j2-1][1] if (j2-1) < len(tok_spans) and j2 > j1 else tok_spans[j1][1]
                spans.append((start, end))
    # Merge overlaps
    spans.sort()
    merged: List[Tuple[int,int]] = []
    for s,e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s,e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged
