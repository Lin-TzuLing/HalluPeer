import os
import json
from pathlib import Path
from typing import List

import yaml
from omegaconf import OmegaConf

try:
    from easydict import EasyDict as edict  # type: ignore
except Exception:
    # Fallback: behave like dict (attribute access not supported, but enough for configs usage)
    def edict(x):
        return x


def get_project_base_dir() -> Path:
    # Assume utils.py is located in hallupeer/ directory, parent.parent points to project root
    return Path(__file__).parent.parent


def load_configs(config_list: List[str]):
    """
    Load multiple YAML files and merge them into a single OmegaConf object.
    """
    base_dir = get_project_base_dir()
    merged_cfg = OmegaConf.create()

    for file_path in config_list:
        abs_path = base_dir / file_path if not os.path.isabs(file_path) else Path(file_path)
        try:
            new_cfg = OmegaConf.load(abs_path)
            merged_cfg = OmegaConf.merge(merged_cfg, new_cfg)
        except Exception as e:
            print(f"Error loading {abs_path}: {e}")

    return merged_cfg



def load_prompt_template(filename: str) -> str:
    base_dir = get_project_base_dir()
    candidate = Path(filename)
    if candidate.is_absolute():
        path = candidate
    else:
        rel = base_dir / filename
        if rel.exists():
            path = rel
        else:
            path = base_dir / "prompts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {filename} (resolved to {path})")
    return path.read_text(encoding="utf-8")


def load_json_configs(file_path: str):
    base_dir = get_project_base_dir()
    abs_path = (
        base_dir / file_path if not os.path.isabs(file_path) else Path(file_path)
    )

    with open(abs_path, "r", encoding="utf-8") as f:
        return json.loads(f.read())
    

def kwargs_to_edict(**kwargs) -> edict:
    """
    Convert keyword arguments to an EasyDict object.
    """
    return edict(kwargs)
