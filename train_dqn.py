from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch

torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim

from agents.neural_agent import NeuralAgent, QNetwork
from agents.replay_buffer import ReplayBuffer, Transition
from game.encoders import ACTION_SIZE, STATE_SIZE, encode_action, encode_state
from game.env import BancoImobiliarioEnv


GAMMA = 0.98
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TARGET_UPDATE_EVERY = 20
SAVE_EVERY = 50
OPTIMIZE_EVERY_STEPS = 200


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    return env.current_player_index


def optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device: str,
):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    transitions = replay_buffer.sample(BATCH_SIZE)

    state_action_batch = torch.tensor(
        np.stack([t.state_action for t in transitions]),
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.tensor(
        [t.reward for t in transitions],
        dtype=torch.float32,
        device=device,
    )

    dones = torch.tensor(
        [t.done for t in transitions],
        dtype=torch.float32,
        device=device,
    )

    current_q = policy_net(state_action_batch)

    # Calcula o max Q das próximas ações de forma vetorizada.
    next_q_values = [0.0] * len(transitions)
    next_state_action_rows = []
    row_to_transition_index = []

    for transition_index, transition in enumerate(transitions):
        if transition.done or not transition.next_actions:
            continue

        next_state_batch = np.repeat(
            transition.next_state[None, :],
            len(transition.next_actions),
            axis=0,
        )
        next_actions_batch = np.stack(transition.next_actions, axis=0)
        next_state_action_batch = np.concatenate(
            [next_state_batch, next_actions_batch],
            axis=1,
        )

        next_state_action_rows.append(next_state_action_batch)
        row_to_transition_index.extend([transition_index] * len(transition.next_actions))

    with torch.no_grad():
        if next_state_action_rows:
            all_next = np.concatenate(next_state_action_rows, axis=0)
            all_next_tensor = torch.tensor(all_next, dtype=torch.float32, device=device)
            all_next_q = target_net(all_next_tensor).detach().cpu().numpy()

            for q_value, transition_index in zip(all_next_q, row_to_transition_index):
                next_q_values[transition_index] = max(
                    next_q_values[transition_index],
                    float(q_value),
                )

    next_q = torch.tensor(next_q_values, dtype=torch.float32, device=device)
    expected_q = rewards + (1.0 - dones) * GAMMA * next_q

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())

def train(episodes: int, seed: int, output_path: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Treinando em: {device}")

    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(seed=seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = max(1, int(episodes * 0.70))

    for episode in range(1, episodes + 1):
        env = BancoImobiliarioEnv(num_players=4, seed=seed + episode)

        epsilon_progress = min(1.0, episode / epsilon_decay_episodes)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epsilon_progress

        agent = NeuralAgent(
            player_id=0,
            model=policy_net,
            epsilon=epsilon,
            device=device,
            seed=seed + episode,
        )

        total_reward = 0.0
        losses = []
        steps = 0

        while not env.done:
            state = env.get_state()
            acting_player = get_acting_player(env)
            agent.player_id = acting_player
            valid_actions = env.get_valid_actions(acting_player)

            if not valid_actions:
                break

            action = agent.choose_action(state, valid_actions)

            state_vec = encode_state(state)
            action_vec = encode_action(action)
            state_action_vec = np.concatenate([state_vec, action_vec], axis=0)

            next_state, reward, done, info = env.step(action)

            next_acting_player = get_acting_player(env)
            next_valid_actions = env.get_valid_actions(next_acting_player)
            next_action_vecs = [encode_action(a) for a in next_valid_actions]
            next_state_vec = encode_state(next_state)

            replay_buffer.push(
                Transition(
                    state_action=state_action_vec,
                    reward=reward,
                    next_state=next_state_vec,
                    next_actions=next_action_vecs,
                    done=done,
                )
            )

            if steps % OPTIMIZE_EVERY_STEPS == 0:
                loss = optimize_model(policy_net, target_net, optimizer, replay_buffer, device)
                if loss is not None:
                    losses.append(loss)

            total_reward += reward
            steps += 1

        if episode % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % SAVE_EVERY == 0 or episode == episodes:
            torch.save(
                {
                    "model_state_dict": policy_net.state_dict(),
                    "episode": episode,
                    "state_size": STATE_SIZE,
                    "action_size": ACTION_SIZE,
                },
                output_path,
            )

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        winner = env.winner
        winner_name = env.players[winner].name if winner is not None else "nenhum"

        print(
            f"Episódio {episode:04d} | "
            f"epsilon={epsilon:.3f} | "
            f"steps={steps:04d} | "
            f"reward={total_reward:8.2f} | "
            f"loss={avg_loss:.5f} | "
            f"vencedor={winner_name}"
        )

    print(f"Modelo salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/dqn_agent.pt")
    args = parser.parse_args()

    train(args.episodes, args.seed, args.output)


if __name__ == "__main__":
    main()
