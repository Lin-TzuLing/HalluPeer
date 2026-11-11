from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Union

def read_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_many_json(paths: Iterable[Union[str, Path]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        data = read_json(p)
        if isinstance(data, list):
            out.extend(data)
        elif isinstance(data, dict):
            out.append(data)
        else:
            raise ValueError(f"Unsupported JSON root type at {p}: {type(data)}")
    return out

def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def list_json_files(input_path: Union[str, Path]) -> List[Path]:
    p = Path(input_path)
    if p.is_dir():
        return sorted([x for x in p.glob("*.json") if x.is_file()])
    if p.is_file():
        return [p]
    raise FileNotFoundError(str(p))
