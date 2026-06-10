from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.neat_agent import (
    NEAT_INPUT_SIZE,
    NeatAgent,
    create_neat_network,
    save_neat_agent_checkpoint,
)
from agents.random_agent import RandomAgent
from game.env import BancoImobiliarioEnv

REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0


@dataclass
class SeatParticipant:
    genome_id: Optional[int]
    agent: Any
    tag: str = ""


@dataclass
class HallOfFameMember:
    name: str
    network: Any
    fitness: float
    generation: int
    path: str


class PureRandomAgent:
    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}
        return self.random.choice(valid_actions)


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
    return env.current_player_index


def rank_seats(env: BancoImobiliarioEnv, net_worths: List[float]) -> List[int]:
    ranked_seats = sorted(
        range(env.num_players),
        key=lambda seat: (
            not env.players[seat].bankrupt,
            net_worths[seat],
        ),
        reverse=True,
    )
    ranks = [0] * env.num_players
    for rank, seat in enumerate(ranked_seats, start=1):
        ranks[seat] = rank
    return ranks


def run_neat_game(
    participants: List[SeatParticipant],
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    env = BancoImobiliarioEnv(num_players=4, seed=seed)
    rewards_by_seat = [0.0] * 4
    steps = 0

    while not env.done and steps < max_steps:
        acting_player = get_acting_player(env)
        valid_actions = env.get_valid_actions(acting_player)

        if not valid_actions:
            if env.recover_no_action_state(acting_player):
                continue
            env.finish_interrupted_game()
            break

        agent = participants[acting_player].agent
        agent.player_id = acting_player
        action = agent.choose_action(env.get_state(), valid_actions)
        _, raw_reward, _, _ = env.step(action)

        rewards_by_seat[acting_player] += max(
            REWARD_CLIP_MIN,
            min(REWARD_CLIP_MAX, raw_reward),
        )
        steps += 1

    if not env.done:
        env.finish_interrupted_game()

    net_worths = [env._reward_net_worth(seat) for seat in range(4)]
    ranks = rank_seats(env, net_worths)

    return {
        "winner": env.winner,
        "steps": steps,
        "net_worths": net_worths,
        "ranks": ranks,
        "rewards_by_seat": rewards_by_seat,
        "bankrupt": [player.bankrupt for player in env.players],
    }


def seat_fitness(result: Dict[str, Any], seat: int) -> float:
    rank = int(result["ranks"][seat])
    net_worth = float(result["net_worths"][seat])
    reward = float(result["rewards_by_seat"][seat])
    fitness = 0.0

    fitness += reward * 0.50
    fitness += net_worth / 8.0
    fitness += (5 - rank) * 150.0

    if result["winner"] == seat:
        fitness += 1000.0
    if result["bankrupt"][seat]:
        fitness -= 250.0

    return fitness


class NeatSelfPlayEvaluator:
    def __init__(
        self,
        games_per_genome: int,
        baseline_games: int,
        hall_of_fame_games: int,
        hall_of_fame_size: int,
        champion_games: int,
        champion_margin: float,
        seed: int,
        max_steps: int,
        output_path: str,
        hall_of_fame_dir: str,
        start_generation: int,
    ):
        self.games_per_genome = games_per_genome
        self.baseline_games = baseline_games
        self.hall_of_fame_games = hall_of_fame_games
        self.hall_of_fame_size = hall_of_fame_size
        self.champion_games = champion_games
        self.champion_margin = champion_margin
        self.seed = seed
        self.max_steps = max_steps
        self.output_path = output_path
        self.hall_of_fame_dir = hall_of_fame_dir
        self.generation = start_generation
        self.best_fitness = float("-inf")
        self.hall_of_fame = self._load_hall_of_fame()
        self.champion = self.hall_of_fame[0] if self.hall_of_fame else None
        if self.champion is not None:
            self.best_fitness = self.champion.fitness
            print(
                f"Campeao atual carregado: {self.champion.name} "
                f"(fitness={self.champion.fitness:.2f})"
            )

    def evaluate_genomes(self, genomes: List[Tuple[int, Any]], config: Any):
        rng = random.Random(self.seed + self.generation)
        networks = {
            genome_id: create_neat_network(genome, config)
            for genome_id, genome in genomes
        }
        fitness_sums = {genome_id: 0.0 for genome_id, _ in genomes}
        games_played = {genome_id: 0 for genome_id, _ in genomes}
        genome_items = list(genomes)
        game_counter = 0
        population_games = 0
        baseline_games = 0
        hall_games = 0

        for round_index in range(self.games_per_genome):
            rng.shuffle(genome_items)

            for start in range(0, len(genome_items), 4):
                group = genome_items[start : start + 4]
                participants: List[SeatParticipant] = []

                for seat, (genome_id, _) in enumerate(group):
                    participants.append(
                        SeatParticipant(
                            genome_id=genome_id,
                            tag="candidate",
                            agent=NeatAgent(
                                player_id=seat,
                                network=networks[genome_id],
                                seed=self.seed + self.generation * 100_000 + round_index * 1000 + seat,
                            ),
                        )
                    )

                while len(participants) < 4:
                    seat = len(participants)
                    participants.append(
                        SeatParticipant(
                            genome_id=None,
                            tag="heuristic",
                            agent=RandomAgent(
                                player_id=seat,
                                seed=self.seed + self.generation * 100_000 + game_counter * 10 + seat,
                            ),
                        )
                    )

                rng.shuffle(participants)
                game_seed = self.seed + self.generation * 100_000 + game_counter
                result = run_neat_game(
                    participants=participants,
                    seed=game_seed,
                    max_steps=self.max_steps,
                )

                for seat, participant in enumerate(participants):
                    if participant.genome_id is None:
                        continue
                    fitness_sums[participant.genome_id] += seat_fitness(result, seat)
                    games_played[participant.genome_id] += 1

                game_counter += 1
                population_games += 1

        for baseline_round in range(self.baseline_games):
            rng.shuffle(genome_items)

            for start in range(0, len(genome_items), 2):
                group = genome_items[start : start + 2]
                participants = []

                for seat, (genome_id, _) in enumerate(group):
                    participants.append(
                        SeatParticipant(
                            genome_id=genome_id,
                            tag="candidate",
                            agent=NeatAgent(
                                player_id=seat,
                                network=networks[genome_id],
                                seed=self.seed
                                + self.generation * 100_000
                                + baseline_round * 10_000
                                + start
                                + seat,
                            ),
                        )
                    )

                participants.append(
                    SeatParticipant(
                        genome_id=None,
                        tag="heuristic",
                        agent=RandomAgent(
                            player_id=len(participants),
                            seed=self.seed
                            + self.generation * 100_000
                            + baseline_round * 10_000
                            + start
                            + 100,
                        ),
                    )
                )
                participants.append(
                    SeatParticipant(
                        genome_id=None,
                        tag="pure_random",
                        agent=PureRandomAgent(
                            player_id=len(participants),
                            seed=self.seed
                            + self.generation * 100_000
                            + baseline_round * 10_000
                            + start
                            + 200,
                        ),
                    )
                )

                rng.shuffle(participants)
                result = run_neat_game(
                    participants=participants,
                    seed=self.seed + self.generation * 100_000 + game_counter,
                    max_steps=self.max_steps,
                )

                for seat, participant in enumerate(participants):
                    if participant.genome_id is None:
                        continue
                    fitness_sums[participant.genome_id] += seat_fitness(result, seat)
                    games_played[participant.genome_id] += 1

                game_counter += 1
                baseline_games += 1

        hall_members = self.hall_of_fame[: self.hall_of_fame_size]
        if hall_members and self.hall_of_fame_games > 0:
            for hall_round in range(self.hall_of_fame_games):
                rng.shuffle(genome_items)

                for start in range(0, len(genome_items), 3):
                    group = genome_items[start : start + 3]
                    participants = []

                    for seat, (genome_id, _) in enumerate(group):
                        participants.append(
                            SeatParticipant(
                                genome_id=genome_id,
                                tag="candidate",
                                agent=NeatAgent(
                                    player_id=seat,
                                    network=networks[genome_id],
                                    seed=self.seed
                                    + self.generation * 100_000
                                    + hall_round * 10_000
                                    + start
                                    + seat,
                                ),
                            )
                        )

                    hall_member = hall_members[(start // 3 + hall_round) % len(hall_members)]
                    participants.append(
                        SeatParticipant(
                            genome_id=None,
                            tag="hall_of_fame",
                            agent=NeatAgent(
                                player_id=len(participants),
                                network=hall_member.network,
                                seed=self.seed + self.generation * 100_000 + hall_round * 10_000 + start,
                            ),
                        )
                    )

                    while len(participants) < 4:
                        seat = len(participants)
                        participants.append(
                            SeatParticipant(
                                genome_id=None,
                                tag="pure_random",
                                agent=PureRandomAgent(
                                    player_id=seat,
                                    seed=self.seed
                                    + self.generation * 100_000
                                    + hall_round * 10_000
                                    + start
                                    + seat,
                                ),
                            )
                        )

                    rng.shuffle(participants)
                    game_seed = self.seed + self.generation * 100_000 + game_counter
                    result = run_neat_game(
                        participants=participants,
                        seed=game_seed,
                        max_steps=self.max_steps,
                    )

                    for seat, participant in enumerate(participants):
                        if participant.genome_id is None:
                            continue
                        fitness_sums[participant.genome_id] += seat_fitness(result, seat)
                        games_played[participant.genome_id] += 1

                    game_counter += 1
                    hall_games += 1

        for genome_id, genome in genomes:
            played = max(1, games_played[genome_id])
            genome.fitness = fitness_sums[genome_id] / float(played)

        best_id, best_genome = max(genomes, key=lambda item: item[1].fitness)
        avg_fitness = float(np.mean([genome.fitness for _, genome in genomes]))
        print(
            f"Geracao {self.generation:04d} | "
            f"jogos={game_counter:04d} | "
            f"pop={population_games:04d} | "
            f"base={baseline_games:04d} | "
            f"hof={hall_games:04d} | "
            f"fitness_med={avg_fitness:8.2f} | "
            f"melhor={best_genome.fitness:8.2f} | "
            f"genoma={best_id}"
        )

        self._maybe_promote_champion(
            genome_id=best_id,
            genome=best_genome,
            config=config,
            rng=rng,
        )

        self.generation += 1

    def _load_hall_of_fame(self) -> List[HallOfFameMember]:
        paths = []
        output = Path(self.output_path)
        if output.is_file():
            paths.append(output)

        hall_dir = Path(self.hall_of_fame_dir)
        if hall_dir.is_dir():
            paths.extend(sorted(hall_dir.glob("*.pkl")))

        members = []
        seen_paths = set()
        for path in paths:
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            try:
                from agents.neat_agent import load_neat_agent_checkpoint

                network, metadata = load_neat_agent_checkpoint(str(path))
            except Exception as exc:
                print(f"Ignorando hall of fame invalido {path}: {exc}")
                continue

            generation = metadata.get("generation")
            fitness = metadata.get("fitness")
            members.append(
                HallOfFameMember(
                    name=path.stem,
                    network=network,
                    fitness=float(fitness) if fitness is not None else float("-inf"),
                    generation=int(generation) if isinstance(generation, int) else -1,
                    path=str(path),
                )
            )

        members.sort(key=lambda member: member.fitness, reverse=True)
        return members[: self.hall_of_fame_size]

    def _maybe_promote_champion(
        self,
        genome_id: int,
        genome: Any,
        config: Any,
        rng: random.Random,
    ):
        candidate_fitness = float(genome.fitness)
        if self.champion is not None:
            passed_gate, candidate_score, champion_score = self._passes_champion_gate(
                genome=genome,
                config=config,
                rng=rng,
            )
            gate_text = (
                f"gate candidato={candidate_score:.2f} "
                f"campeao={champion_score:.2f} "
                f"margem={self.champion_margin:.2f}"
            )
            if not passed_gate:
                print(f"Campeao mantido; genoma {genome_id} nao passou no {gate_text}.")
                return
            print(f"Genoma {genome_id} passou no {gate_text}.")

        self.best_fitness = candidate_fitness
        promoted_genome = copy.deepcopy(genome)
        save_neat_agent_checkpoint(
            path=self.output_path,
            genome=promoted_genome,
            config=config,
            generation=self.generation,
            fitness=candidate_fitness,
        )

        hall_path = self._hall_of_fame_path(candidate_fitness)
        save_neat_agent_checkpoint(
            path=hall_path,
            genome=promoted_genome,
            config=config,
            generation=self.generation,
            fitness=candidate_fitness,
        )

        promoted_member = HallOfFameMember(
            name=Path(hall_path).stem,
            network=create_neat_network(promoted_genome, config),
            fitness=candidate_fitness,
            generation=self.generation,
            path=hall_path,
        )
        self.champion = promoted_member
        self.hall_of_fame.insert(0, promoted_member)
        self.hall_of_fame.sort(key=lambda member: member.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[: self.hall_of_fame_size]

        print(
            f"Novo campeao NEAT salvo em: {self.output_path} "
            f"e arquivado em {hall_path}"
        )

    def _passes_champion_gate(
        self,
        genome: Any,
        config: Any,
        rng: random.Random,
    ) -> Tuple[bool, float, float]:
        if self.champion is None:
            return True, 0.0, 0.0

        candidate_network = create_neat_network(genome, config)
        candidate_score = 0.0
        champion_score = 0.0

        for game_index in range(self.champion_games):
            participants = [
                SeatParticipant(
                    genome_id=None,
                    tag="candidate",
                    agent=NeatAgent(
                        player_id=0,
                        network=candidate_network,
                        seed=self.seed + self.generation * 1_000_000 + game_index,
                    ),
                ),
                SeatParticipant(
                    genome_id=None,
                    tag="champion",
                    agent=NeatAgent(
                        player_id=1,
                        network=self.champion.network,
                        seed=self.seed + self.generation * 1_000_000 + game_index + 10_000,
                    ),
                ),
                SeatParticipant(
                    genome_id=None,
                    tag="heuristic",
                    agent=RandomAgent(
                        player_id=2,
                        seed=self.seed + self.generation * 1_000_000 + game_index + 20_000,
                    ),
                ),
                SeatParticipant(
                    genome_id=None,
                    tag="pure_random",
                    agent=PureRandomAgent(
                        player_id=3,
                        seed=self.seed + self.generation * 1_000_000 + game_index + 30_000,
                    ),
                ),
            ]
            rng.shuffle(participants)
            result = run_neat_game(
                participants=participants,
                seed=self.seed + self.generation * 1_000_000 + game_index,
                max_steps=self.max_steps,
            )

            for seat, participant in enumerate(participants):
                if participant.tag == "candidate":
                    candidate_score += seat_fitness(result, seat)
                elif participant.tag == "champion":
                    champion_score += seat_fitness(result, seat)

        games = max(1, self.champion_games)
        candidate_avg = candidate_score / float(games)
        champion_avg = champion_score / float(games)
        return candidate_avg > champion_avg + self.champion_margin, candidate_avg, champion_avg

    def _hall_of_fame_path(self, fitness: float) -> str:
        os.makedirs(self.hall_of_fame_dir, exist_ok=True)
        safe_fitness = int(round(fitness))
        return str(
            Path(self.hall_of_fame_dir)
            / f"neat_hof_gen_{self.generation:06d}_fit_{safe_fitness}.pkl"
        )


def load_config(config_path: str, pop_size: Optional[int]):
    import neat

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    if int(config.genome_config.num_inputs) != NEAT_INPUT_SIZE:
        raise ValueError(
            f"Config NEAT invalida: num_inputs={config.genome_config.num_inputs}, "
            f"esperado={NEAT_INPUT_SIZE}"
        )
    if pop_size is not None:
        config.pop_size = max(4, int(pop_size))
    return config


def train(
    config_path: str,
    generations: int,
    games_per_genome: int,
    baseline_games: int,
    seed: int,
    output_path: str,
    checkpoint_prefix: str,
    checkpoint_every: int,
    max_steps: int,
    resume: Optional[str],
    hall_of_fame_dir: str,
    hall_of_fame_size: int,
    hall_of_fame_games: int,
    champion_games: int,
    champion_margin: float,
    pop_size: Optional[int],
):
    import neat

    random.seed(seed)
    np.random.seed(seed)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.dirname(checkpoint_prefix)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if hall_of_fame_dir:
        os.makedirs(hall_of_fame_dir, exist_ok=True)

    if resume:
        population = neat.Checkpointer.restore_checkpoint(resume)
        config = population.config
    else:
        config = load_config(config_path, pop_size)
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    if checkpoint_every > 0:
        population.add_reporter(
            neat.Checkpointer(
                generation_interval=checkpoint_every,
                filename_prefix=checkpoint_prefix,
            )
        )

    evaluator = NeatSelfPlayEvaluator(
        games_per_genome=max(1, games_per_genome),
        baseline_games=max(0, baseline_games),
        hall_of_fame_games=max(0, hall_of_fame_games),
        hall_of_fame_size=max(1, hall_of_fame_size),
        champion_games=max(1, champion_games),
        champion_margin=champion_margin,
        seed=seed,
        max_steps=max_steps,
        output_path=output_path,
        hall_of_fame_dir=hall_of_fame_dir,
        start_generation=int(getattr(population, "generation", 0)),
    )

    winner = population.run(evaluator.evaluate_genomes, generations)
    print(
        f"Treino finalizado. Melhor genoma da populacao tem fitness={float(winner.fitness):.2f}. "
        f"Campeao salvo protegido em: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="neat_raw_config.ini")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--games-per-genome", type=int, default=2)
    parser.add_argument("--baseline-games", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/best_neat_raw_agent.pkl")
    parser.add_argument("--checkpoint-prefix", type=str, default="models/neat_checkpoints/neat-")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--hall-of-fame-dir", type=str, default="models/neat_hall_of_fame")
    parser.add_argument("--hall-of-fame-size", type=int, default=6)
    parser.add_argument("--hall-of-fame-games", type=int, default=1)
    parser.add_argument("--champion-games", type=int, default=16)
    parser.add_argument("--champion-margin", type=float, default=25.0)
    args = parser.parse_args()

    train(
        config_path=args.config,
        generations=max(1, args.generations),
        games_per_genome=max(1, args.games_per_genome),
        baseline_games=max(0, args.baseline_games),
        seed=args.seed,
        output_path=args.output,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_every=args.checkpoint_every,
        max_steps=max(1, args.max_steps),
        resume=args.resume,
        hall_of_fame_dir=args.hall_of_fame_dir,
        hall_of_fame_size=args.hall_of_fame_size,
        hall_of_fame_games=args.hall_of_fame_games,
        champion_games=args.champion_games,
        champion_margin=args.champion_margin,
        pop_size=args.pop_size,
    )


if __name__ == "__main__":
    main()
