import copy
import os
import random
import time
from typing import Any, Dict, List

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import torch

torch.set_num_threads(1)

from agents.neural_agent import NeuralAgent, QNetwork
from game.env import BancoImobiliarioEnv
from ui.pygame_view import PygameView

STEP_DELAY_SECONDS = 0.25
MIN_STEP_DELAY_SECONDS = 0.03
MAX_STEP_DELAY_SECONDS = 2.0
START_PAUSED = True
FIXED_SEED_ENV = "BANCO_SEED"
MODEL_PATH = "models/best_dqn_agent.pt"


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


def make_env_and_agents():
    game_seed = make_game_seed()
    print(f"Seed da partida: {game_seed}")
    env = BancoImobiliarioEnv(num_players=4, seed=game_seed, enable_undo=True)
    model = QNetwork()

    agents = [NeuralAgent(player_id=i, model=model, epsilon=0.02, seed=game_seed + i) for i in range(4)]

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Carregando modelo treinado: {MODEL_PATH}")
            agents[0].load(MODEL_PATH)
        except RuntimeError as exc:
            print("Não foi possível carregar o modelo antigo.")
            print("Isso é esperado se você treinou antes da mecânica de casas/hotel.")
            print("Treine novamente com: python train_dqn.py --episodes 300")
            print(f"Detalhe técnico: {exc}")
            agents = [NeuralAgent(player_id=i, model=model, epsilon=0.25, seed=game_seed + i) for i in range(4)]
    else:
        print("Modelo treinado não encontrado. Rodando rede não treinada.")
        print("Para treinar: python train_dqn.py --episodes 300")
        agents = [NeuralAgent(player_id=i, model=model, epsilon=0.25, seed=game_seed + i) for i in range(4)]

    # Todos os agentes compartilham a mesma rede.
    # Isso representa self-play com uma política única sendo usada por todos os jogadores.
    shared_model = agents[0].model
    for agent in agents:
        agent.model = shared_model
        agent.model.eval()

    return env, agents


def run_one_ai_action(
    env: BancoImobiliarioEnv,
    agents: List[NeuralAgent],
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
