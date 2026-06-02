import os
import time

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
SEED = 42
MODEL_PATH = "models/dqn_agent.pt"


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
    return env.current_player_index


def make_env_and_agents():
    env = BancoImobiliarioEnv(num_players=4, seed=SEED, enable_undo=True)
    model = QNetwork()

    agents = [NeuralAgent(player_id=i, model=model, epsilon=0.02, seed=SEED + i) for i in range(4)]

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Carregando modelo treinado: {MODEL_PATH}")
            agents[0].load(MODEL_PATH)
        except RuntimeError as exc:
            print("Não foi possível carregar o modelo antigo.")
            print("Isso é esperado se você treinou antes da mecânica de casas/hotel.")
            print("Treine novamente com: python train_dqn.py --episodes 300")
            print(f"Detalhe técnico: {exc}")
            agents = [NeuralAgent(player_id=i, model=model, epsilon=0.25, seed=SEED + i) for i in range(4)]
    else:
        print("Modelo treinado não encontrado. Rodando rede não treinada.")
        print("Para treinar: python train_dqn.py --episodes 300")
        agents = [NeuralAgent(player_id=i, model=model, epsilon=0.25, seed=SEED + i) for i in range(4)]

    # Todos os agentes compartilham a mesma rede.
    # Isso representa self-play com uma política única sendo usada por todos os jogadores.
    shared_model = agents[0].model
    for agent in agents:
        agent.model = shared_model
        agent.model.eval()

    return env, agents


def run_one_ai_action(env, agents):
    if env.done:
        return

    state = env.get_state()
    acting_player = get_acting_player(env)
    valid_actions = env.get_valid_actions(acting_player)

    if valid_actions:
        action = agents[acting_player].choose_action(state, valid_actions)
        env.step(action)


def main():
    env, agents = make_env_and_agents()
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
            paused = START_PAUSED
            last_step_time = 0.0

        if commands["undo"]:
            if env.undo_last_action():
                paused = True
            last_step_time = time.time()

        if commands["step_once"]:
            run_one_ai_action(env, agents)
            last_step_time = time.time()

        now = time.time()

        if not paused and not env.done and now - last_step_time >= step_delay:
            run_one_ai_action(env, agents)
            last_step_time = now

        view.draw(env, paused=paused, step_delay=step_delay)

    view.quit()


if __name__ == "__main__":
    main()
