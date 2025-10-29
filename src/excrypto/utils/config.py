import yaml, json, hashlib
from pathlib import Path

def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    return cfg

def cfg_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:8]
