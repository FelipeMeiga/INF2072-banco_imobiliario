import os
import json
import time
from pathlib import Path

import torch

from game import encoders


def find_checkpoints(paths):
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pt")))
            files.extend(sorted(path.glob("*.pkl")))
        elif path.is_file():
            files.append(path)
    return files


def inspect_checkpoint(path: Path):
    try:
        data = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}

    info = {"path": str(path)}
    info["algorithm"] = data.get("algorithm", "dqn") if isinstance(data, dict) else "unknown"
    info["state_size"] = int(data.get("state_size")) if isinstance(data, dict) and data.get("state_size") is not None else None
    info["action_size"] = int(data.get("action_size")) if isinstance(data, dict) and data.get("action_size") is not None else None
    info["episode"] = int(data.get("episode")) if isinstance(data, dict) and data.get("episode") is not None else None
    return info


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=["models", "models/checkpoints"]) 
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    state_size = encoders.STATE_SIZE
    action_size = encoders.ACTION_SIZE

    checkpoint_files = find_checkpoints(args.paths)
    results = {
        "checked_at": time.time(),
        "state_size": state_size,
        "action_size": action_size,
        "files": [],
    }

    for p in checkpoint_files:
        info = inspect_checkpoint(p)
        compat = None
        if info.get("state_size") is not None and info.get("action_size") is not None:
            compat = (info["state_size"] == state_size) and (info["action_size"] == action_size)
        info["compatible_with_current_encoders"] = compat
        results["files"].append(info)

    out_path = args.out
    if out_path is None:
        ts = int(time.time())
        out_path = f"results/compatibility_{ts}.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote compatibility results to {out_path}")


if __name__ == "__main__":
    main()
