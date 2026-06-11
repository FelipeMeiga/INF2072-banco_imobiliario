import os
import json
import time
from pathlib import Path
from typing import Dict

from tournament import discover_checkpoints, load_competitors, run_tournament


def aggregate_stats(dest: Dict, src: Dict):
    for name, stats in src.items():
        if name not in dest:
            dest[name] = {
                "games": 0,
                "wins": 0,
                "total_rank": 0,
                "total_net_worth": 0.0,
                "total_turns": 0,
                "action_counts": {},
            }
        d = dest[name]
        d["games"] += int(stats.get("games", 0))
        d["wins"] += int(stats.get("wins", 0))
        d["total_rank"] += int(stats.get("total_rank", 0))
        d["total_net_worth"] += float(stats.get("total_net_worth", 0.0))
        d["total_turns"] += int(stats.get("total_turns", 0))
        for k, v in stats.get("action_counts", {}).items():
            d["action_counts"][k] = d["action_counts"].get(k, 0) + int(v)


def main():
    import argparse
    import random
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["models/checkpoints"]) 
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = discover_checkpoints(args.checkpoints)
    competitors = load_competitors(checkpoint_paths, device=device)
    if not competitors:
        raise SystemExit("No competitors found for tournament.")

    total_games = args.games
    batch = max(1, int(args.batch_size))
    batches = (total_games + batch - 1) // batch

    aggregated = {}
    cur_seed = args.seed
    for i in range(batches):
        this_batch_games = batch if (i < batches - 1) else (total_games - batch * (batches - 1))
        print(f"Running batch {i+1}/{batches}: games={this_batch_games}")
        stats = run_tournament(competitors=competitors, games=this_batch_games, seed=cur_seed, device=device)
        aggregate_stats(aggregated, {name: {
            "games": s.games,
            "wins": s.wins,
            "total_rank": s.total_rank,
            "total_net_worth": s.total_net_worth,
            "total_turns": s.total_turns,
            "action_counts": s.action_counts,
        } for name, s in stats.items()})
        cur_seed += this_batch_games

    out_path = args.out or f"results/tournament_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"games": total_games, "seed": args.seed, "device": device}, "stats": aggregated}, fh, indent=2)

    print(f"Wrote tournament results to {out_path}")


if __name__ == "__main__":
    main()
