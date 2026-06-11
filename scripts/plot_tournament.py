import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def championship_score(stats):
    games = stats.get("games", 1)
    wins = stats.get("wins", 0)
    total_rank = stats.get("total_rank", 0)
    total_net = stats.get("total_net_worth", 0.0)
    win_rate = wins / games if games else 0.0
    avg_rank = total_rank / games if games else 0.0
    avg_net = total_net / games if games else 0.0
    return (win_rate * 10000.0) - (avg_rank * 100.0) + (avg_net / 10.0)


def top_n(stats_dict, n=10):
    scored = [(name, championship_score(s), s) for name, s in stats_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def plot_winrate_bar(scored, out_path):
    names = [s[0] for s in scored]
    win_rates = [ (s[2].get('wins',0)/s[2].get('games',1))*100.0 for s in scored]
    plt.figure(figsize=(10,6))
    plt.barh(names[::-1], win_rates[::-1])
    plt.xlabel('Win rate (%)')
    plt.title('Top competitors by championship score: Win rate')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_bar(scored, out_path):
    names = [s[0] for s in scored]
    scores = [s[1] for s in scored]
    plt.figure(figsize=(10,6))
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel('Championship score')
    plt.title('Top competitors by championship score')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    stats = data.get("stats", data)
    os.makedirs(args.out_dir, exist_ok=True)
    scored = top_n(stats, n=args.top)

    plot_winrate_bar(scored, os.path.join(args.out_dir, "winrate_top.png"))
    plot_score_bar(scored, os.path.join(args.out_dir, "score_top.png"))

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
