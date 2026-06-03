import copy
import os
import random
import time
from typing import Any, Dict, List

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import torch

torch.set_num_threads(1)

from agents.neural_agent import NeuralAgent, QNetwork
from agents.neat_agent import NeatAgent, load_neat_agent_checkpoint
from agents.ppo_agent import PPOActorCritic, PPOAgent
from game.encoders import (
    DEFAULT_ENCODER,
    ENCODER_RAW,
    get_action_size,
    get_state_size,
    normalize_encoder_name,
)
from game.env import BancoImobiliarioEnv
from ui.pygame_view import PygameView

STEP_DELAY_SECONDS = 0.25
MIN_STEP_DELAY_SECONDS = 0.03
MAX_STEP_DELAY_SECONDS = 2.0
START_PAUSED = True
FIXED_SEED_ENV = "BANCO_SEED"
MODEL_PATH_ENV = "BANCO_MODEL_PATH"
ENCODER_ENV = "BANCO_ENCODER"
NEAT_MODEL_PATH = "models/best_neat_raw_agent.pkl"
PPO_RAW_MODEL_PATH = "models/best_ppo_raw_agent.pt"
PPO_MODEL_PATH = "models/best_ppo_agent.pt"
DQN_MODEL_PATH = "models/best_dqn_agent.pt"


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
    return env.current_player_index


def make_game_seed() -> int:
    fixed_seed = os.environ.get(FIXED_SEED_ENV)
    if fixed_seed is not None:
        return int(fixed_seed)
    return random.SystemRandom().randint(0, 2**31 - 1)


def resolve_model_path() -> str | None:
    explicit_path = os.environ.get(MODEL_PATH_ENV)
    if explicit_path:
        return explicit_path
    if os.path.exists(NEAT_MODEL_PATH):
        return NEAT_MODEL_PATH
    if os.path.exists(PPO_RAW_MODEL_PATH):
        return PPO_RAW_MODEL_PATH
    if os.path.exists(PPO_MODEL_PATH):
        return PPO_MODEL_PATH
    if os.path.exists(DQN_MODEL_PATH):
        return DQN_MODEL_PATH
    return None


def make_untrained_agents():
    encoder_name = normalize_encoder_name(os.environ.get(ENCODER_ENV, ENCODER_RAW))
    model = PPOActorCritic(
        state_size=get_state_size(encoder_name),
        action_size=get_action_size(encoder_name),
    )
    return [
        PPOAgent(player_id=i, model=model, deterministic=False, encoder=encoder_name)
        for i in range(4)
    ]


def load_agents(model_path: str, game_seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path.lower().endswith(".pkl"):
        network, metadata = load_neat_agent_checkpoint(model_path)
        agents = [
            NeatAgent(player_id=i, network=network, seed=game_seed + i)
            for i in range(4)
        ]
        generation = metadata.get("generation")
        label = "NEAT/raw"
        if isinstance(generation, int):
            label = f"{label}/gen{generation}"
        return agents, label

    checkpoint = torch.load(model_path, map_location=device)
    algorithm = checkpoint.get("algorithm", "dqn")

    if algorithm == "ppo":
        encoder_name = normalize_encoder_name(checkpoint.get("encoder", DEFAULT_ENCODER))
        expected_state_size = get_state_size(encoder_name)
        expected_action_size = get_action_size(encoder_name)
        state_size = int(checkpoint.get("state_size", expected_state_size))
        action_size = int(checkpoint.get("action_size", expected_action_size))
        if state_size != expected_state_size or action_size != expected_action_size:
            raise ValueError(
                "Checkpoint PPO incompativel com o estado/acao atual: "
                f"salvo=({state_size}, {action_size}), "
                f"esperado=({expected_state_size}, {expected_action_size})"
            )
        hidden_size = int(checkpoint.get("hidden_size", 512))
        model = PPOActorCritic(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        agents = [
            PPOAgent(
                player_id=i,
                model=model,
                device=device,
                deterministic=True,
                encoder=encoder_name,
            )
            for i in range(4)
        ]
        return agents, f"PPO/{encoder_name}"

    model = QNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    agents = [
        NeuralAgent(player_id=i, model=model, epsilon=0.02, device=device, seed=game_seed + i)
        for i in range(4)
    ]
    return agents, "DQN"


def make_env_and_agents():
    game_seed = make_game_seed()
    print(f"Seed da partida: {game_seed}")
    env = BancoImobiliarioEnv(num_players=4, seed=game_seed, enable_undo=True)

    model_path = resolve_model_path()
    if model_path is not None and os.path.exists(model_path):
        try:
            agents, algorithm = load_agents(model_path, game_seed)
            print(f"Carregando modelo {algorithm}: {model_path}")
        except (RuntimeError, KeyError, ValueError) as exc:
            print("Nao foi possivel carregar o modelo salvo.")
            print("Isso e esperado apos mudancas no estado, nas acoes ou na arquitetura.")
            print("Treine novamente com: py train_ppo.py --episodes 300")
            print(f"Detalhe tecnico: {exc}")
            agents = make_untrained_agents()
    else:
        print("Modelo treinado nao encontrado. Rodando PPO nao treinado.")
        print("Para treinar: py train_ppo.py --episodes 300")
        agents = make_untrained_agents()

    return env, agents


def run_one_ai_action(
    env: BancoImobiliarioEnv,
    agents: List[Any],
    replay_actions: List[Dict[str, Any]],
    replay_cursor: int,
) -> int:
    if env.done:
        return replay_cursor

    acting_player = get_acting_player(env)
    valid_actions = env.get_valid_actions(acting_player)

    if not valid_actions:
        return replay_cursor

    if replay_cursor < len(replay_actions):
        action = replay_actions[replay_cursor]
    else:
        state = env.get_state()
        action = agents[acting_player].choose_action(state, valid_actions)
        replay_actions.append(copy.deepcopy(action))

    env.step(copy.deepcopy(action))
    return replay_cursor + 1


def main():
    env, agents = make_env_and_agents()
    replay_actions: List[Dict[str, Any]] = []
    replay_cursor = 0
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
            env, agents = make_env_and_agents()
            replay_actions = []
            replay_cursor = 0
            paused = START_PAUSED
            last_step_time = 0.0

        if commands["undo"]:
            if env.undo_last_action():
                replay_cursor = max(0, replay_cursor - 1)
                paused = True
            last_step_time = time.time()

        if commands["step_once"]:
            replay_cursor = run_one_ai_action(env, agents, replay_actions, replay_cursor)
            last_step_time = time.time()

        now = time.time()

        if not paused and not env.done and now - last_step_time >= step_delay:
            replay_cursor = run_one_ai_action(env, agents, replay_actions, replay_cursor)
            last_step_time = now

        view.draw(env, paused=paused, step_delay=step_delay)

    view.quit()


if __name__ == "__main__":
    main()
