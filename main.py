import time

from agents.random_agent import RandomAgent
from game.env import BancoImobiliarioEnv
from ui.pygame_view import PygameView

STEP_DELAY_SECONDS = 0.35
SEED = 42


def main():
    env = BancoImobiliarioEnv(num_players=4, seed=SEED)
    agents = [RandomAgent(player_id=i, seed=SEED + i) for i in range(4)]
    view = PygameView()

    running = True
    last_step_time = 0.0

    while running:
        running = view.handle_events()

        now = time.time()

        if not env.done and now - last_step_time >= STEP_DELAY_SECONDS:
            state = env.get_state()

            # Quando existe troca pendente, quem decide é o jogador alvo.
            if env.pending_trade is not None:
                acting_player = env.pending_trade["to"]
            else:
                acting_player = env.current_player_index

            valid_actions = env.get_valid_actions(acting_player)

            if valid_actions:
                action = agents[acting_player].choose_action(state, valid_actions)
                env.step(action)

            last_step_time = now

        view.draw(env)

    view.quit()


if __name__ == "__main__":
    main()
