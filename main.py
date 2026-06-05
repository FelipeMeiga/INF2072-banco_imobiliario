import copy
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIXED_SEED_ENV = "BANCO_SEED"
MODEL_PATH_ENV = "BANCO_MODEL_PATH"
ENCODER_ENV = "BANCO_ENCODER"
PPO_DETERMINISTIC_ENV = "BANCO_PPO_DETERMINISTIC"
NEAT_MODEL_PATH = "models/best_neat_raw_agent.pkl"
PPO_RAW_MODEL_PATH = "models/best_ppo_raw_agent.pt"
PPO_MODEL_PATH = "models/best_ppo_agent.pt"
DQN_MODEL_PATH = "models/best_dqn_agent.pt"
REPLAY_DIR = "replays"
LATEST_REPLAY_PATH = os.path.join(REPLAY_DIR, "latest_replay.json")
REPLAY_PATH_ENV = "BANCO_REPLAY_PATH"
REPLAY_SAVE_PATH_ENV = "BANCO_REPLAY_SAVE_PATH"
REPLAY_SAVE_INTERVAL_ACTIONS = 25


def resolve_existing_path(path: str) -> str | None:
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(PROJECT_ROOT, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


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
        resolved = resolve_existing_path(explicit_path)
        if resolved is not None:
            return resolved
        print(f"Modelo indicado em {MODEL_PATH_ENV} nao encontrado: {explicit_path}")
        print("Tentando carregar o melhor modelo padrao disponivel.")

    for candidate in (
        PPO_RAW_MODEL_PATH,
        PPO_MODEL_PATH,
        NEAT_MODEL_PATH,
        DQN_MODEL_PATH,
    ):
        resolved = resolve_existing_path(candidate)
        if resolved is not None:
            return resolved
    return None


def use_deterministic_ppo() -> bool:
    value = os.environ.get(PPO_DETERMINISTIC_ENV, "").strip().lower()
    return value in {"1", "true", "sim", "yes", "y"}


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
        deterministic = use_deterministic_ppo()
        agents = [
            PPOAgent(
                player_id=i,
                model=model,
                device=device,
                deterministic=deterministic,
                encoder=encoder_name,
            )
            for i in range(4)
        ]
        mode = "deterministico" if deterministic else "estocastico"
        return agents, f"PPO/{encoder_name}/{mode}"

    model = QNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    agents = [
        NeuralAgent(player_id=i, model=model, epsilon=0.02, device=device, seed=game_seed + i)
        for i in range(4)
    ]
    return agents, "DQN"


def make_env_and_agents(game_seed: Optional[int] = None):
    if game_seed is None:
        game_seed = make_game_seed()
    print(f"Seed da partida: {game_seed}")
    env = BancoImobiliarioEnv(num_players=4, seed=game_seed, enable_undo=True)

    model_path = resolve_model_path()
    algorithm = "untrained"
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

    metadata = {
        "seed": game_seed,
        "model_path": model_path,
        "algorithm": algorithm,
    }
    return env, agents, metadata


def make_replay_save_path(seed: int) -> str:
    explicit_path = os.environ.get(REPLAY_SAVE_PATH_ENV)
    if explicit_path:
        return explicit_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(REPLAY_DIR, f"replay_{timestamp}_seed_{seed}.json")


def make_replay_payload(
    seed: int,
    metadata: Dict[str, Any],
    replay_actions: List[Dict[str, Any]],
    replay_cursor: int,
    completed: bool,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "created_by": "main.py",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "model_path": metadata.get("model_path"),
        "algorithm": metadata.get("algorithm"),
        "actions": replay_actions,
        "cursor": replay_cursor,
        "completed": completed,
    }


def write_json_atomic(path: str, payload: Dict[str, Any], required: bool = True) -> bool:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    try:
        os.replace(temp_path, path)
        return True
    except PermissionError as exc:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        if required:
            raise
        print(f"Nao foi possivel atualizar {path}: {exc}")
        return False


def save_replay_file(
    path: str,
    seed: int,
    metadata: Dict[str, Any],
    replay_actions: List[Dict[str, Any]],
    replay_cursor: int,
    completed: bool,
    update_latest: bool = True,
):
    payload = make_replay_payload(seed, metadata, replay_actions, replay_cursor, completed)
    write_json_atomic(path, payload)
    if update_latest and os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(LATEST_REPLAY_PATH)):
        write_json_atomic(LATEST_REPLAY_PATH, payload, required=False)


def load_replay_file(path: str) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    seed = int(payload["seed"])
    actions = list(payload.get("actions", []))
    metadata = {
        "seed": seed,
        "model_path": payload.get("model_path"),
        "algorithm": f"replay/{payload.get('algorithm', 'unknown')}",
        "source_path": path,
    }
    return seed, actions, metadata


def capture_replay_snapshot(env: BancoImobiliarioEnv) -> Dict[str, Any]:
    return env._capture_snapshot()


def restore_replay_snapshot(env: BancoImobiliarioEnv, snapshot: Dict[str, Any]):
    env._restore_snapshot(snapshot)
    env.undo_history = []


def build_replay_snapshots(seed: int, replay_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    replay_env = BancoImobiliarioEnv(num_players=4, seed=seed, enable_undo=False)
    snapshots = [capture_replay_snapshot(replay_env)]
    for action in replay_actions:
        if replay_env.done:
            break
        replay_env.step(copy.deepcopy(action))
        snapshots.append(capture_replay_snapshot(replay_env))
    return snapshots


def seek_replay(
    env: BancoImobiliarioEnv,
    replay_snapshots: List[Dict[str, Any]],
    replay_cursor: int,
) -> int:
    if not replay_snapshots:
        return 0
    replay_cursor = max(0, min(replay_cursor, len(replay_snapshots) - 1))
    restore_replay_snapshot(env, replay_snapshots[replay_cursor])
    return replay_cursor


def run_one_ai_action(
    env: BancoImobiliarioEnv,
    agents: List[Any],
    replay_actions: List[Dict[str, Any]],
    replay_cursor: int,
    replay_snapshots: List[Dict[str, Any]],
    replay_locked: bool,
) -> Tuple[int, bool]:
    if replay_cursor < len(replay_snapshots) - 1:
        restore_replay_snapshot(env, replay_snapshots[replay_cursor + 1])
        return replay_cursor + 1, False

    if env.done:
        return replay_cursor, False

    acting_player = get_acting_player(env)
    valid_actions = env.get_valid_actions(acting_player)

    if not valid_actions:
        return replay_cursor, False

    if replay_cursor < len(replay_actions):
        action = replay_actions[replay_cursor]
    else:
        if replay_locked:
            return replay_cursor, False
        state = env.get_state()
        action = agents[acting_player].choose_action(state, valid_actions)
        replay_actions.append(copy.deepcopy(action))
        replay_snapshots[:] = replay_snapshots[: replay_cursor + 1]

    env.step(copy.deepcopy(action))
    replay_snapshots.append(capture_replay_snapshot(env))
    return replay_cursor + 1, replay_cursor >= len(replay_actions) - 1


def main():
    replay_load_path = os.environ.get(REPLAY_PATH_ENV)
    replay_locked = False

    if replay_load_path:
        replay_seed, replay_actions, replay_metadata = load_replay_file(replay_load_path)
        env, agents, metadata = make_env_and_agents(replay_seed)
        metadata.update(replay_metadata)
        replay_snapshots = build_replay_snapshots(replay_seed, replay_actions)
        replay_cursor = seek_replay(env, replay_snapshots, 0)
        replay_save_path = replay_load_path
        replay_locked = True
        print(f"Replay carregado: {replay_load_path} ({len(replay_actions)} acoes)")
    else:
        env, agents, metadata = make_env_and_agents()
        replay_actions: List[Dict[str, Any]] = []
        replay_snapshots = [capture_replay_snapshot(env)]
        replay_cursor = 0
        replay_save_path = make_replay_save_path(int(metadata["seed"]))
        print(f"Gravando replay em: {replay_save_path}")

    view = PygameView()
    last_saved_action_count = 0

    running = True
    paused = START_PAUSED
    step_delay = STEP_DELAY_SECONDS
    last_step_time = 0.0

    while running:
        new_action = False
        commands = view.handle_events()
        running = commands["running"]

        if commands["toggle_pause"]:
            paused = not paused

        if commands["speed_up"]:
            step_delay = max(MIN_STEP_DELAY_SECONDS, step_delay / 1.5)

        if commands["speed_down"]:
            step_delay = min(MAX_STEP_DELAY_SECONDS, step_delay * 1.5)

        if commands["save_replay"] and not replay_locked:
            save_replay_file(
                replay_save_path,
                int(metadata["seed"]),
                metadata,
                replay_actions,
                replay_cursor,
                env.done,
            )
            last_saved_action_count = len(replay_actions)
            env.last_message = f"Replay salvo em {replay_save_path}."

        if commands["load_latest"]:
            if os.path.exists(LATEST_REPLAY_PATH):
                replay_seed, replay_actions, replay_metadata = load_replay_file(LATEST_REPLAY_PATH)
                env, agents, metadata = make_env_and_agents(replay_seed)
                metadata.update(replay_metadata)
                replay_snapshots = build_replay_snapshots(replay_seed, replay_actions)
                replay_cursor = seek_replay(env, replay_snapshots, 0)
                replay_save_path = LATEST_REPLAY_PATH
                replay_locked = True
                paused = True
                last_step_time = 0.0
                print(f"Replay carregado: {LATEST_REPLAY_PATH} ({len(replay_actions)} acoes)")
            else:
                env.last_message = "Nenhum replay salvo encontrado em replays/latest_replay.json."

        if commands["reset"]:
            env, agents, metadata = make_env_and_agents()
            replay_actions = []
            replay_snapshots = [capture_replay_snapshot(env)]
            replay_cursor = 0
            replay_save_path = make_replay_save_path(int(metadata["seed"]))
            replay_locked = False
            last_saved_action_count = 0
            paused = START_PAUSED
            last_step_time = 0.0
            print(f"Gravando replay em: {replay_save_path}")

        if commands["seek_index"] is not None:
            replay_cursor = seek_replay(env, replay_snapshots, int(commands["seek_index"]))
            paused = True
            last_step_time = time.time()

        if commands["undo"]:
            replay_cursor = seek_replay(env, replay_snapshots, replay_cursor - 1)
            paused = True
            last_step_time = time.time()

        if commands["step_once"]:
            replay_cursor, generated_action = run_one_ai_action(
                env,
                agents,
                replay_actions,
                replay_cursor,
                replay_snapshots,
                replay_locked,
            )
            new_action = new_action or generated_action
            last_step_time = time.time()

        now = time.time()

        if not paused and not env.done and now - last_step_time >= step_delay:
            replay_cursor, generated_action = run_one_ai_action(
                env,
                agents,
                replay_actions,
                replay_cursor,
                replay_snapshots,
                replay_locked,
            )
            new_action = new_action or generated_action
            last_step_time = now

        if (
            not replay_locked
            and (new_action or env.done)
            and (
                len(replay_actions) - last_saved_action_count >= REPLAY_SAVE_INTERVAL_ACTIONS
                or env.done
            )
        ):
            save_replay_file(
                replay_save_path,
                int(metadata["seed"]),
                metadata,
                replay_actions,
                replay_cursor,
                env.done,
            )
            last_saved_action_count = len(replay_actions)

        view.draw(
            env,
            paused=paused,
            step_delay=step_delay,
            replay_cursor=replay_cursor,
            replay_total=len(replay_actions),
            replay_path=replay_save_path,
            replay_locked=replay_locked,
        )

    if not replay_locked and replay_actions:
        save_replay_file(
            replay_save_path,
            int(metadata["seed"]),
            metadata,
            replay_actions,
            replay_cursor,
            env.done,
        )
    view.quit()


if __name__ == "__main__":
    main()
