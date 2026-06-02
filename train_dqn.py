from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim

from agents.neural_agent import NeuralAgent, QNetwork
from agents.replay_buffer import ReplayBuffer, Transition
from game.encoders import ACTION_SIZE, STATE_SIZE, encode_action, encode_state
from game.env import BancoImobiliarioEnv
from tournament import discover_checkpoints, load_competitors, print_scoreboard, run_tournament


GAMMA = 0.98
BATCH_SIZE = 128
LEARNING_RATE = 3e-5
TARGET_UPDATE_EVERY = 50
SAVE_EVERY = 50
OPTIMIZE_EVERY_STEPS = 20
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0
OPPONENT_Q_WEIGHT = 0.75


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
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

            best_next_q_by_transition: Dict[int, float] = {}
            for q_value, transition_index in zip(all_next_q, row_to_transition_index):
                best_next_q_by_transition[transition_index] = max(
                    best_next_q_by_transition.get(transition_index, float("-inf")),
                    float(q_value),
                )

            for transition_index, best_next_q in best_next_q_by_transition.items():
                transition = transitions[transition_index]
                if transition.next_acting_player == transition.acting_player:
                    next_q_values[transition_index] = best_next_q
                else:
                    next_q_values[transition_index] = -OPPONENT_Q_WEIGHT * best_next_q

    next_q = torch.tensor(next_q_values, dtype=torch.float32, device=device)
    expected_q = rewards + (1.0 - dones) * GAMMA * next_q

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())

def save_checkpoint(policy_net: QNetwork, episode: int, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "episode": episode,
            "state_size": STATE_SIZE,
            "action_size": ACTION_SIZE,
        },
        output_path,
    )


def run_checkpoint_championship(
    checkpoint_dir: str,
    games: int,
    seed: int,
    device: str,
    latest: int,
) -> Optional[Tuple[str, float, str]]:
    checkpoint_paths = discover_checkpoints([checkpoint_dir])
    competitors = load_competitors(
        checkpoint_paths,
        device=device,
        limit_latest=latest if latest > 0 else None,
    )

    if not competitors:
        print("Campeonato ignorado: nenhum checkpoint compativel encontrado.")
        return None

    stats = run_tournament(
        competitors=competitors,
        games=games,
        seed=seed,
        device=device,
        include_baselines=True,
    )
    print_scoreboard(stats, limit=10)

    competitor_by_name = {
        competitor.name: competitor
        for competitor in competitors
        if competitor.path is not None
    }
    ranked_dqn = sorted(
        (
            (name, competitor_stats.championship_score, competitor_by_name[name].path)
            for name, competitor_stats in stats.items()
            if name in competitor_by_name
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked_dqn[0] if ranked_dqn else None


def train(
    episodes: int,
    seed: int,
    output_path: str,
    checkpoint_dir: Optional[str],
    checkpoint_every: int,
    tournament_every: int,
    tournament_games: int,
    tournament_latest: int,
    best_output_path: str,
):
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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = max(1, int(episodes * 0.40))
    best_championship_score = float("-inf")

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
                if env.recover_no_action_state(acting_player):
                    continue

                env.finish_interrupted_game()
                break

            action = agent.choose_action(state, valid_actions)

            state_vec = encode_state(state)
            action_vec = encode_action(action)
            state_action_vec = np.concatenate([state_vec, action_vec], axis=0)

            next_state, raw_reward, done, info = env.step(action)
            reward = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, raw_reward))

            next_acting_player = get_acting_player(env)
            next_valid_actions = env.get_valid_actions(next_acting_player)
            next_action_vecs = [encode_action(a) for a in next_valid_actions]
            next_state_vec = encode_state(next_state)

            replay_buffer.push(
                Transition(
                    state_action=state_action_vec,
                    acting_player=acting_player,
                    reward=reward,
                    next_state=next_state_vec,
                    next_actions=next_action_vecs,
                    next_acting_player=next_acting_player,
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

        should_save_main = episode % SAVE_EVERY == 0 or episode == episodes
        should_save_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_every > 0
            and (episode % checkpoint_every == 0 or episode == episodes)
        )

        if should_save_main:
            save_checkpoint(policy_net, episode, output_path)

        if should_save_checkpoint:
            checkpoint_path = str(Path(checkpoint_dir) / f"dqn_ep_{episode:06d}.pt")
            save_checkpoint(policy_net, episode, checkpoint_path)

        if (
            checkpoint_dir is not None
            and tournament_every > 0
            and tournament_games > 0
            and (episode % tournament_every == 0 or episode == episodes)
        ):
            print("")
            print(f"=== Campeonato de checkpoints no episodio {episode:04d} ===")
            championship_winner = run_checkpoint_championship(
                checkpoint_dir=checkpoint_dir,
                games=tournament_games,
                seed=seed + 100_000 + episode,
                device=device,
                latest=tournament_latest,
            )
            if championship_winner is not None:
                best_name, best_score, best_path = championship_winner
                if best_score > best_championship_score:
                    best_championship_score = best_score
                    best_dir = os.path.dirname(best_output_path)
                    if best_dir:
                        os.makedirs(best_dir, exist_ok=True)
                    shutil.copyfile(best_path, best_output_path)
                    print(
                        f"Novo melhor checkpoint: {best_name} "
                        f"(score={best_score:.2f}) salvo em {best_output_path}"
                    )
            print("=== Fim do campeonato ===")
            print("")

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
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--tournament-every", type=int, default=0)
    parser.add_argument("--tournament-games", type=int, default=24)
    parser.add_argument("--tournament-latest", type=int, default=8)
    parser.add_argument("--best-output", type=str, default="models/best_dqn_agent.pt")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        tournament_every=args.tournament_every,
        tournament_games=args.tournament_games,
        tournament_latest=args.tournament_latest,
        best_output_path=args.best_output,
    )


if __name__ == "__main__":
    main()
