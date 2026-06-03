import os
import random
import time
from typing import Any, List, Tuple

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from game.github_monopoly import GithubStyleMonopolyEnv
from train_neat_github import create_network, load_github_neat_checkpoint
from ui.pygame_view import PygameView

STEP_DELAY_SECONDS = 0.25
MIN_STEP_DELAY_SECONDS = 0.03
MAX_STEP_DELAY_SECONDS = 2.0
START_PAUSED = True
FIXED_SEED_ENV = "BANCO_SEED"
MODEL_PATH_ENV = "BANCO_GITHUB_MODEL_PATH"
GITHUB_MODEL_PATH = "models/neat_github_checkpoints/neat-github-3.pkl"


class DefaultGithubNetwork:
    """Fallback policy so the visualizer opens even before training."""

    def activate(self, inputs: List[float]) -> List[float]:
        return [
            0.85,  # buy
            0.75,  # pay jail
            0.00,  # mortgage
            0.80,  # unmortgage
            0.18,  # auction bid, roughly 720
            0.75,  # build houses
            0.00,  # sell houses
            0.10,  # offer trade
            0.55,  # accept trade
        ]


def make_game_seed() -> int:
    fixed_seed = os.environ.get(FIXED_SEED_ENV)
    if fixed_seed is not None:
        return int(fixed_seed)
    return random.SystemRandom().randint(0, 2**31 - 1)


def resolve_model_path() -> str | None:
    explicit_path = os.environ.get(MODEL_PATH_ENV)
    if explicit_path:
        return explicit_path
    if os.path.exists(GITHUB_MODEL_PATH):
        return GITHUB_MODEL_PATH
    return None


def load_networks() -> Tuple[List[Any], str, bool]:
    model_path = resolve_model_path()
    if model_path and os.path.exists(model_path):
        network, metadata = load_github_neat_checkpoint(model_path)
        include_money_context = bool(metadata.get("include_money_context", False))
        generation = metadata.get("generation")
        label = "NEAT/GitHub-style"
        if isinstance(generation, int):
            label = f"{label}/gen{generation}"
        networks = [network]
        networks.extend(
            create_network(metadata["genome"], metadata["config"])
            for _ in range(3)
        )
        return networks, label, include_money_context

    fallback = DefaultGithubNetwork()
    return [fallback for _ in range(4)], "fallback/GitHub-style", False


def make_env():
    game_seed = make_game_seed()
    networks, label, include_money_context = load_networks()
    print(f"Seed da partida GitHub-style: {game_seed}")
    print(f"Carregando visualizacao: {label}")
    env = GithubStyleMonopolyEnv(
        networks,
        seed=game_seed,
        include_money_context=include_money_context,
        enable_undo=True,
        visual_mode=True,
    )
    return env


def main():
    env = make_env()
    view = PygameView()

    running = True
    paused = START_PAUSED
    step_delay = STEP_DELAY_SECONDS
    last_step_time = 0.0

    while running:
        commands = view.handle_events()
        running = commands["running"]

        if commands["toggle_pause"]:
            paused = not paused

        if commands["speed_up"]:
            step_delay = max(MIN_STEP_DELAY_SECONDS, step_delay / 1.5)

        if commands["speed_down"]:
            step_delay = min(MAX_STEP_DELAY_SECONDS, step_delay * 1.5)

        if commands["reset"]:
            env = make_env()
            paused = START_PAUSED
            last_step_time = 0.0

        if commands["undo"]:
            if env.undo_last_action():
                paused = True
            last_step_time = time.time()

        if commands["step_once"] and not env.done:
            env.step()
            last_step_time = time.time()

        now = time.time()
        if not paused and not env.done and now - last_step_time >= step_delay:
            env.step()
            last_step_time = now

        view.draw(env, paused=paused, step_delay=step_delay)

    view.quit()


if __name__ == "__main__":
    main()
