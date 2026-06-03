from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from game.github_monopoly import (
    GITHUB_INPUT_SIZE,
    GITHUB_OUTPUT_SIZE,
    GithubStyleMonopolyEnv,
    Outcome,
)

BRACKET_FITNESS_WEIGHT = 100.0
SCORE_RATE_FITNESS_WEIGHT = 10.0


@dataclass
class GithubContestant:
    genome_id: int
    genome: Any
    network: Any


class TrainingTimeLimitReached(Exception):
    def __init__(self, checkpoint_path: str):
        super().__init__(checkpoint_path)
        self.checkpoint_path = checkpoint_path


class TimeLimitReporter:
    def __init__(self, max_seconds: float, filename_prefix: str):
        self.max_seconds = max_seconds
        self.filename_prefix = filename_prefix
        self.started_at = time.time()
        self.current_generation = 0

    def start_generation(self, generation: int):
        self.current_generation = generation

    def end_generation(self, config: Any, population: Any, species_set: Any):
        if time.time() - self.started_at < self.max_seconds:
            return

        import neat

        next_generation = self.current_generation + 1
        checkpoint_path = f"{self.filename_prefix}{next_generation}"
        neat.Checkpointer(
            generation_interval=None,
            filename_prefix=self.filename_prefix,
        ).save_checkpoint(config, population, species_set, next_generation)
        raise TrainingTimeLimitReached(checkpoint_path)

    def post_evaluate(self, config: Any, population: Any, species: Any, best_genome: Any):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config: Any, generation: int, best: Any):
        pass

    def species_stagnant(self, sid: int, species: Any):
        pass

    def info(self, msg: str):
        pass


def create_network(genome: Any, config: Any) -> Any:
    import neat

    return neat.nn.FeedForwardNetwork.create(genome, config)


def save_github_neat_checkpoint(
    path: str,
    genome: Any,
    config: Any,
    generation: int,
    fitness: float,
    champion_score: float,
    champion_bracket: int,
    include_money_context: bool,
):
    payload = {
        "algorithm": "neat_github",
        "input_size": GITHUB_INPUT_SIZE + (1 if include_money_context else 0),
        "output_size": GITHUB_OUTPUT_SIZE,
        "generation": generation,
        "fitness": fitness,
        "champion_score": champion_score,
        "champion_bracket": champion_bracket,
        "fitness_formula": "bracket * 100 + score_rate * 10",
        "bracket_fitness_weight": BRACKET_FITNESS_WEIGHT,
        "score_rate_fitness_weight": SCORE_RATE_FITNESS_WEIGHT,
        "include_money_context": include_money_context,
        "genome": genome,
        "config": config,
    }
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_github_neat_checkpoint(path: str) -> Tuple[Any, Dict[str, Any]]:
    with open(path, "rb") as file:
        payload = pickle.load(file)

    if payload.get("algorithm") != "neat_github":
        raise ValueError(f"Checkpoint nao e NEAT GitHub-style: {path}")

    network = create_network(payload["genome"], payload["config"])
    return network, payload


class GithubTournamentEvaluator:
    """
    Evaluates genomes like the GitHub MonopolyNEAT tournament.

    The population is shuffled into groups of four. Each group plays several
    full games in the GitHub-style environment. The best scorer in each group
    advances one bracket. At the end, genome fitness is relative to the previous
    champion bracket, matching the original repo's champion-score idea.
    """

    def __init__(
        self,
        games_per_bracket: int,
        seed: int,
        output_path: str,
        hall_of_fame_dir: str,
        checkpoint_every_generation: bool,
        include_money_context: bool,
        start_generation: int,
    ):
        self.games_per_bracket = max(1, games_per_bracket)
        self.seed = seed
        self.output_path = output_path
        self.hall_of_fame_dir = hall_of_fame_dir
        self.checkpoint_every_generation = checkpoint_every_generation
        self.include_money_context = include_money_context
        self.generation = start_generation
        self.champion_score = 0.0
        self.champion_bracket = 0
        self.champion_genome: Optional[Any] = None
        self._load_existing_champion()

    def evaluate_genomes(self, genomes: List[Tuple[int, Any]], config: Any):
        rng = random.Random(self.seed + self.generation)
        contestants = [
            GithubContestant(
                genome_id=genome_id,
                genome=genome,
                network=create_network(genome, config),
            )
            for genome_id, genome in genomes
        ]
        id_by_network = {id(contestant.network): contestant.genome_id for contestant in contestants}
        genome_by_id = {genome_id: genome for genome_id, genome in genomes}
        brackets = {genome_id: 0 for genome_id, _ in genomes}
        total_scores = {genome_id: 0.0 for genome_id, _ in genomes}
        games_by_genome = {genome_id: 0 for genome_id, _ in genomes}
        total_games = 0
        round_index = 0

        while len(contestants) > 1:
            rng.shuffle(contestants)
            next_round: List[GithubContestant] = []
            round_games = 0

            for start in range(0, len(contestants), 4):
                group = contestants[start : start + 4]
                if len(group) < 4:
                    for contestant in group:
                        brackets[contestant.genome_id] += 1
                    next_round.extend(group)
                    continue

                group_scores = {contestant.genome_id: 0.0 for contestant in group}
                for game_index in range(self.games_per_bracket):
                    game_seed = (
                        self.seed
                        + self.generation * 10_000_000
                        + round_index * 1_000_000
                        + start * 10_000
                        + game_index
                    )
                    env = GithubStyleMonopolyEnv(
                        [contestant.network for contestant in group],
                        seed=game_seed,
                        include_money_context=self.include_money_context,
                    )
                    outcome = env.play()

                    for contestant in group:
                        games_by_genome[contestant.genome_id] += 1

                    if outcome == Outcome.DRAW:
                        for player in env.players:
                            genome_id = id_by_network[id(player.brain.network)]
                            group_scores[genome_id] += 0.25
                            total_scores[genome_id] += 0.25
                    else:
                        winner_seat = env.winner_index(outcome)
                        if winner_seat is not None:
                            winner_network = env.players[winner_seat].brain.network
                            genome_id = id_by_network[id(winner_network)]
                            group_scores[genome_id] += 1.0
                            total_scores[genome_id] += 1.0

                    total_games += 1
                    round_games += 1

                best_score = max(group_scores.values())
                tied = [
                    contestant
                    for contestant in group
                    if group_scores[contestant.genome_id] == best_score
                ]
                winner = rng.choice(tied)
                brackets[winner.genome_id] += 1
                next_round.append(winner)

            print(
                f"Geracao {self.generation:04d} | round={round_index:02d} | "
                f"competidores={len(contestants):04d} | jogos={round_games:05d} | "
                f"avancam={len(next_round):04d}"
            )
            contestants = next_round
            round_index += 1

        for genome_id, genome in genomes:
            played = max(1, games_by_genome[genome_id])
            score_rate = total_scores[genome_id] / float(played)
            genome.fitness = (
                brackets[genome_id] * BRACKET_FITNESS_WEIGHT
                + score_rate * SCORE_RATE_FITNESS_WEIGHT
            )

        champion = contestants[0]
        champion_fitness = float(genome_by_id[champion.genome_id].fitness)
        champion_score_rate = total_scores[champion.genome_id] / float(
            max(1, games_by_genome[champion.genome_id])
        )

        avg_fitness = sum(float(genome.fitness) for _, genome in genomes) / max(1, len(genomes))
        best_wins = total_scores[champion.genome_id]
        print(
            f"Geracao {self.generation:04d} | jogos={total_games:05d} | "
            f"fitness_med={avg_fitness:8.2f} | campeao={champion.genome_id} | "
            f"bracket={brackets[champion.genome_id]} | wins_score={best_wins:.2f} | "
            f"score_rate={champion_score_rate:.3f} | fitness={champion_fitness:.2f}"
        )

        self._maybe_save_champion(
            genome_id=champion.genome_id,
            genome=genome_by_id[champion.genome_id],
            bracket=brackets[champion.genome_id],
            fitness=champion_fitness,
            config=config,
        )
        self.generation += 1

    def _load_existing_champion(self):
        path = Path(self.output_path)
        if not path.is_file():
            return
        try:
            _, metadata = load_github_neat_checkpoint(str(path))
        except Exception as exc:
            print(f"Ignorando checkpoint GitHub-style existente: {exc}")
            return
        self.champion_score = float(metadata.get("champion_score", metadata.get("fitness", 0.0)))
        self.champion_bracket = int(metadata.get("champion_bracket", 0))
        self.champion_genome = metadata.get("genome")
        print(
            f"Campeao GitHub-style carregado: "
            f"score={self.champion_score:.2f}, bracket={self.champion_bracket}"
        )

    def _maybe_save_champion(
        self,
        genome_id: int,
        genome: Any,
        bracket: int,
        fitness: float,
        config: Any,
    ):
        if self.champion_genome is not None and fitness <= self.champion_score:
            print(
                f"Campeao salvo mantido; genoma {genome_id} "
                f"fitness={fitness:.2f} <= melhor={self.champion_score:.2f}."
            )
            return

        self.champion_genome = copy.deepcopy(genome)
        self.champion_bracket = bracket
        self.champion_score = fitness
        self._save_champion(
            genome=self.champion_genome,
            config=config,
            fitness=fitness,
        )
        print(
            f"Novo melhor campeao GitHub-style: genoma {genome_id} "
            f"fitness={fitness:.2f} salvo em {self.output_path}"
        )

    def _save_champion(self, genome: Any, config: Any, fitness: float):
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        save_github_neat_checkpoint(
            path=self.output_path,
            genome=genome,
            config=config,
            generation=self.generation,
            fitness=fitness,
            champion_score=self.champion_score,
            champion_bracket=self.champion_bracket,
            include_money_context=self.include_money_context,
        )

        if not self.checkpoint_every_generation:
            return

        os.makedirs(self.hall_of_fame_dir, exist_ok=True)
        archive_path = (
            Path(self.hall_of_fame_dir)
            / f"github_neat_gen_{self.generation:06d}_fit_{int(round(fitness))}.pkl"
        )
        save_github_neat_checkpoint(
            path=str(archive_path),
            genome=genome,
            config=config,
            generation=self.generation,
            fitness=fitness,
            champion_score=self.champion_score,
            champion_bracket=self.champion_bracket,
            include_money_context=self.include_money_context,
        )


def load_config(config_path: str, pop_size: Optional[int], include_money_context: bool):
    import neat

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    expected_inputs = GITHUB_INPUT_SIZE + (1 if include_money_context else 0)
    if int(config.genome_config.num_inputs) != expected_inputs:
        raise ValueError(
            f"Config NEAT invalida: num_inputs={config.genome_config.num_inputs}, "
            f"esperado={expected_inputs}"
        )
    if int(config.genome_config.num_outputs) != GITHUB_OUTPUT_SIZE:
        raise ValueError(
            f"Config NEAT invalida: num_outputs={config.genome_config.num_outputs}, "
            f"esperado={GITHUB_OUTPUT_SIZE}"
        )
    if pop_size is not None:
        config.pop_size = max(4, int(pop_size))
    if config.pop_size % 4 != 0:
        raise ValueError("O torneio GitHub-style exige pop_size divisivel por 4.")
    return config


def mutate_initial_population(population: Any):
    for genome in population.population.values():
        genome.mutate(population.config.genome_config)
    population.species.speciate(
        population.config,
        population.population,
        population.generation,
    )


def train(
    config_path: str,
    generations: int,
    games_per_bracket: int,
    seed: int,
    output_path: str,
    checkpoint_prefix: str,
    checkpoint_every: int,
    hall_of_fame_dir: str,
    resume: Optional[str],
    pop_size: Optional[int],
    mutate_initial: bool,
    save_every_generation: bool,
    include_money_context: bool,
    max_hours: float,
):
    import neat

    random.seed(seed)

    if resume:
        population = neat.Checkpointer.restore_checkpoint(resume)
        expected_inputs = GITHUB_INPUT_SIZE + (1 if include_money_context else 0)
        if int(population.config.genome_config.num_inputs) != expected_inputs:
            raise ValueError(
                f"Checkpoint incompativel: num_inputs="
                f"{population.config.genome_config.num_inputs}, esperado={expected_inputs}"
            )
        if int(population.config.genome_config.num_outputs) != GITHUB_OUTPUT_SIZE:
            raise ValueError(
                f"Checkpoint incompativel: num_outputs="
                f"{population.config.genome_config.num_outputs}, esperado={GITHUB_OUTPUT_SIZE}"
            )
    else:
        config = load_config(config_path, pop_size, include_money_context)
        population = neat.Population(config)
        if mutate_initial:
            mutate_initial_population(population)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.dirname(checkpoint_prefix)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    if checkpoint_every > 0:
        population.add_reporter(
            neat.Checkpointer(
                generation_interval=checkpoint_every,
                filename_prefix=checkpoint_prefix,
            )
        )
    if max_hours > 0:
        population.add_reporter(
            TimeLimitReporter(
                max_seconds=max_hours * 3600.0,
                filename_prefix=checkpoint_prefix,
            )
        )

    print(
        "Treinando NEAT GitHub-style | "
        f"inputs={population.config.genome_config.num_inputs} | "
        f"outputs={population.config.genome_config.num_outputs} | "
        f"pop={population.config.pop_size} | "
        f"games_per_bracket={games_per_bracket}"
    )

    evaluator = GithubTournamentEvaluator(
        games_per_bracket=games_per_bracket,
        seed=seed,
        output_path=output_path,
        hall_of_fame_dir=hall_of_fame_dir,
        checkpoint_every_generation=save_every_generation,
        include_money_context=include_money_context,
        start_generation=int(getattr(population, "generation", 0)),
    )

    try:
        winner = population.run(evaluator.evaluate_genomes, max(1, generations))
    except TrainingTimeLimitReached as exc:
        print(
            "Tempo limite atingido no fim da geracao. "
            f"Checkpoint de continuacao salvo em: {exc.checkpoint_path}"
        )
        print(f"Para continuar: py train_neat_github.py --resume {exc.checkpoint_path}")
        return

    print(
        f"Treino finalizado. Melhor genoma da populacao fitness={float(winner.fitness):.2f}. "
        f"Campeao salvo em: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="neat_github_config.ini")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--games-per-bracket", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/best_neat_github_agent.pkl")
    parser.add_argument("--checkpoint-prefix", type=str, default="models/neat_github_checkpoints/neat-github-")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--hall-of-fame-dir", type=str, default="models/neat_github_hall_of_fame")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--no-mutate-initial", action="store_true")
    parser.add_argument("--save-every-generation", action="store_true")
    parser.add_argument("--include-money-context", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="Para automaticamente apos esse tempo, no fim de uma geracao, salvando checkpoint para --resume.",
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        generations=args.generations,
        games_per_bracket=max(1, args.games_per_bracket),
        seed=args.seed,
        output_path=args.output,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_every=args.checkpoint_every,
        hall_of_fame_dir=args.hall_of_fame_dir,
        resume=args.resume,
        pop_size=args.pop_size,
        mutate_initial=not args.no_mutate_initial,
        save_every_generation=args.save_every_generation,
        include_money_context=args.include_money_context,
        max_hours=max(0.0, args.max_hours),
    )


if __name__ == "__main__":
    main()
