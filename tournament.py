from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

torch.set_num_threads(1)

from agents.neural_agent import NeuralAgent, QNetwork
from agents.neat_agent import NeatAgent, load_neat_agent_checkpoint
from agents.ppo_agent import PPOActorCritic, PPOAgent
from agents.random_agent import RandomAgent
from game.encoders import (
    ACTION_SIZE,
    DEFAULT_ENCODER,
    STATE_SIZE,
    get_action_size,
    get_state_size,
    normalize_encoder_name,
)
from game.env import BancoImobiliarioEnv


@dataclass
class Competitor:
    name: str
    kind: str
    path: Optional[str] = None
    model: Optional[object] = None
    episode: Optional[int] = None
    encoder: str = DEFAULT_ENCODER


class PureRandomAgent:
    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, object], valid_actions: List[Dict[str, object]]) -> Dict[str, object]:
        if not valid_actions:
            return {"type": "no_action"}
        return self.random.choice(valid_actions)


@dataclass
class CompetitorStats:
    games: int = 0
    wins: int = 0
    total_rank: int = 0
    total_net_worth: float = 0.0
    total_turns: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)

    def add_action_counts(self, counts: Dict[str, int]):
        for action_type, count_value in counts.items():
            self.action_counts[action_type] = self.action_counts.get(action_type, 0) + count_value

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_rank(self) -> float:
        return self.total_rank / self.games if self.games else 0.0

    @property
    def avg_net_worth(self) -> float:
        return self.total_net_worth / self.games if self.games else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.games if self.games else 0.0

    @property
    def championship_score(self) -> float:
        # Normalized score: win rate dominates, rank breaks ties, net worth gives
        # a smaller signal. This keeps comparisons fair when sampled players have
        # different game counts.
        return (self.win_rate * 10_000.0) - (self.avg_rank * 100.0) + (self.avg_net_worth / 10.0)


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
    return env.current_player_index


def discover_checkpoints(paths: Iterable[str]) -> List[str]:
    checkpoint_paths: List[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            checkpoint_paths.extend(str(item) for item in sorted(path.glob("*.pt")))
            checkpoint_paths.extend(str(item) for item in sorted(path.glob("*.pkl")))
        elif path.is_file():
            checkpoint_paths.append(str(path))

    return sorted(dict.fromkeys(checkpoint_paths))


def load_checkpoint_competitor(path: str, device: str) -> Optional[Competitor]:
    if path.lower().endswith(".pkl"):
        try:
            network, metadata = load_neat_agent_checkpoint(path)
        except Exception as exc:
            print(f"Ignorando checkpoint NEAT invalido {path}: {exc}")
            return None

        generation = metadata.get("generation")
        name = f"neat_raw_gen{generation:04d}" if isinstance(generation, int) else Path(path).stem
        return Competitor(
            name=name,
            kind="neat",
            path=path,
            model=network,
            episode=generation if isinstance(generation, int) else None,
            encoder="raw",
        )

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"Ignorando checkpoint invalido {path}: {exc}")
        return None

    algorithm = checkpoint.get("algorithm", "dqn")
    if algorithm == "ppo":
        try:
            encoder_name = normalize_encoder_name(checkpoint.get("encoder", DEFAULT_ENCODER))
        except ValueError as exc:
            print(f"Ignorando checkpoint PPO incompativel {path}: {exc}")
            return None

        expected_state_size = get_state_size(encoder_name)
        expected_action_size = get_action_size(encoder_name)
        state_size = int(checkpoint.get("state_size", expected_state_size))
        action_size = int(checkpoint.get("action_size", expected_action_size))
        if state_size != expected_state_size or action_size != expected_action_size:
            print(
                f"Ignorando checkpoint PPO incompativel {path}: "
                f"encoder={encoder_name}, salvos=({state_size}, {action_size}), "
                f"esperados=({expected_state_size}, {expected_action_size})"
            )
            return None

        hidden_size = int(checkpoint.get("hidden_size", 512))
        model = PPOActorCritic(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(device)
        kind = "ppo"
    else:
        state_size = checkpoint.get("state_size")
        action_size = checkpoint.get("action_size")
        if state_size != STATE_SIZE or action_size != ACTION_SIZE:
            print(
                f"Ignorando checkpoint DQN incompativel {path}: "
                f"state/action salvos=({state_size}, {action_size}), atuais=({STATE_SIZE}, {ACTION_SIZE})"
            )
            return None
        encoder_name = DEFAULT_ENCODER
        model = QNetwork().to(device)
        kind = "dqn"

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episode = checkpoint.get("episode")
    prefix = "ppo" if kind == "ppo" else "dqn"
    if kind == "ppo" and encoder_name != DEFAULT_ENCODER:
        prefix = f"{prefix}_{encoder_name}"
    suffix = f"{prefix}_ep{episode:04d}" if isinstance(episode, int) else Path(path).stem
    return Competitor(
        name=suffix,
        kind=kind,
        path=path,
        model=model,
        episode=episode if isinstance(episode, int) else None,
        encoder=encoder_name,
    )


def make_agent(competitor: Competitor, seat: int, seed: int, device: str):
    if competitor.kind == "pure_random":
        return PureRandomAgent(player_id=seat, seed=seed)

    if competitor.kind == "heuristic":
        return RandomAgent(player_id=seat, seed=seed)

    if competitor.kind == "ppo":
        assert competitor.model is not None
        return PPOAgent(
            player_id=seat,
            model=competitor.model,
            device=device,
            deterministic=True,
            encoder=competitor.encoder,
        )

    if competitor.kind == "neat":
        assert competitor.model is not None
        return NeatAgent(
            player_id=seat,
            network=competitor.model,
            seed=seed,
        )

    assert competitor.model is not None
    return NeuralAgent(
        player_id=seat,
        model=competitor.model,
        epsilon=0.0,
        device=device,
        seed=seed,
    )


def choose_competitors_for_game(
    competitors: List[Competitor],
    rng: random.Random,
    game_index: int,
) -> List[Competitor]:
    if len(competitors) >= 4:
        return rng.sample(competitors, 4)

    selected = list(competitors)
    filler_counter = count(1)
    while len(selected) < 4:
        selected.append(Competitor(name=f"pure_random_filler_{next(filler_counter)}", kind="pure_random"))
    rng.shuffle(selected)
    return selected


def run_game(
    seated_competitors: List[Competitor],
    seed: int,
    device: str,
    max_steps: int = 20_000,
) -> Dict[str, object]:
    env = BancoImobiliarioEnv(num_players=4, seed=seed)
    agents = [
        make_agent(competitor, seat=seat, seed=seed + seat + 1, device=device)
        for seat, competitor in enumerate(seated_competitors)
    ]
    action_counts_by_seat: List[Dict[str, int]] = [dict() for _ in range(4)]
    steps = 0

    while not env.done and steps < max_steps:
        acting_player = get_acting_player(env)
        valid_actions = env.get_valid_actions(acting_player)

        if not valid_actions:
            if env.recover_no_action_state(acting_player):
                continue
            env.finish_interrupted_game()
            break

        agent = agents[acting_player]
        agent.player_id = acting_player
        action = agent.choose_action(env.get_state(), valid_actions)
        env.step(action)

        action_type = action.get("type", "unknown")
        seat_counts = action_counts_by_seat[acting_player]
        seat_counts[action_type] = seat_counts.get(action_type, 0) + 1
        steps += 1

    if not env.done:
        env.finish_interrupted_game()

    net_worths = [env._reward_net_worth(seat) for seat in range(4)]
    ranked_seats = sorted(
        range(4),
        key=lambda seat: (
            not env.players[seat].bankrupt,
            net_worths[seat],
        ),
        reverse=True,
    )
    ranks = [0] * 4
    for rank, seat in enumerate(ranked_seats, start=1):
        ranks[seat] = rank

    return {
        "winner": env.winner,
        "turns": env.turn_count,
        "steps": steps,
        "net_worths": net_worths,
        "ranks": ranks,
        "action_counts_by_seat": action_counts_by_seat,
    }


def run_tournament(
    competitors: List[Competitor],
    games: int,
    seed: int = 123,
    device: Optional[str] = None,
    include_baselines: bool = True,
) -> Dict[str, CompetitorStats]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    tournament_competitors = list(competitors)

    if include_baselines:
        tournament_competitors.append(Competitor(name="pure_random_baseline", kind="pure_random"))
        tournament_competitors.append(Competitor(name="heuristic_baseline", kind="heuristic"))

    if not tournament_competitors:
        raise ValueError("Nenhum competidor disponivel para o campeonato.")

    stats: Dict[str, CompetitorStats] = {
        competitor.name: CompetitorStats()
        for competitor in tournament_competitors
    }

    for game_index in range(1, games + 1):
        seated = choose_competitors_for_game(tournament_competitors, rng, game_index)
        result = run_game(
            seated_competitors=seated,
            seed=seed + game_index,
            device=device,
        )
        winner = result["winner"]
        ranks = result["ranks"]
        net_worths = result["net_worths"]
        action_counts_by_seat = result["action_counts_by_seat"]

        for seat, competitor in enumerate(seated):
            competitor_stats = stats.setdefault(competitor.name, CompetitorStats())
            competitor_stats.games += 1
            competitor_stats.total_rank += int(ranks[seat])
            competitor_stats.total_net_worth += float(net_worths[seat])
            competitor_stats.total_turns += int(result["turns"])
            competitor_stats.add_action_counts(action_counts_by_seat[seat])

            if winner == seat:
                competitor_stats.wins += 1

    return stats


def print_scoreboard(stats: Dict[str, CompetitorStats], limit: Optional[int] = None):
    rows = sorted(
        stats.items(),
        key=lambda item: item[1].championship_score,
        reverse=True,
    )
    if limit is not None:
        rows = rows[:limit]

    print("Campeonato de checkpoints")
    print("rank | competidor       | jogos | vitorias | win%   | pos_med | patrimonio | score")
    for rank, (name, competitor_stats) in enumerate(rows, start=1):
        print(
            f"{rank:04d} | "
            f"{name[:16]:16s} | "
            f"{competitor_stats.games:5d} | "
            f"{competitor_stats.wins:8d} | "
            f"{competitor_stats.win_rate * 100:5.1f} | "
            f"{competitor_stats.avg_rank:7.2f} | "
            f"{competitor_stats.avg_net_worth:10.2f} | "
            f"{competitor_stats.championship_score:8.2f}"
        )


def load_competitors(checkpoint_paths: List[str], device: str, limit_latest: Optional[int] = None) -> List[Competitor]:
    if limit_latest is not None and limit_latest > 0:
        checkpoint_paths = checkpoint_paths[-limit_latest:]

    competitors = []
    for path in checkpoint_paths:
        competitor = load_checkpoint_competitor(path, device=device)
        if competitor is not None:
            competitors.append(competitor)
    return competitors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["models/checkpoints"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--latest", type=int, default=0)
    parser.add_argument("--no-baselines", "--no-random", dest="no_baselines", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = discover_checkpoints(args.checkpoints)
    competitors = load_competitors(
        checkpoint_paths,
        device=device,
        limit_latest=args.latest if args.latest > 0 else None,
    )

    stats = run_tournament(
        competitors=competitors,
        games=args.games,
        seed=args.seed,
        device=device,
        include_baselines=not args.no_baselines,
    )
    print_scoreboard(stats)


if __name__ == "__main__":
    main()
