from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.ppo_agent import PPOActorCritic, PPOAgent
from game.board import GROUPS
from game.encoders import (
    ENCODER_NAMES,
    ENCODER_RAW,
    get_action_size,
    get_state_size,
    normalize_encoder_name,
)
from game.env import BancoImobiliarioEnv
from tournament import discover_checkpoints, load_competitors, print_scoreboard, run_tournament


GAMMA = 0.98
GAE_LAMBDA = 0.95
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
PPO_EPOCHS = 2
CLIP_EPSILON = 0.20
VALUE_COEF = 0.50
ENTROPY_COEF = 0.02
MAX_GRAD_NORM = 0.50
SAVE_EVERY = 50
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0
UPDATE_EVERY_EPISODES = 4
TRADE_CURRICULUM_RATIO = 0.35


@dataclass
class PPOTransition:
    state_vec: np.ndarray
    action_vecs: np.ndarray
    action_index: int
    old_log_prob: float
    value: float
    reward: float
    done: bool


def get_acting_player(env: BancoImobiliarioEnv) -> int:
    if env.pending_trade is not None:
        return env.pending_trade["to"]
    if env.auction is not None:
        return env.auction["current_bidder"]
    return env.current_player_index


def save_checkpoint(
    model: PPOActorCritic,
    episode: int,
    output_path: str,
    encoder_name: str,
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "algorithm": "ppo",
            "encoder": encoder_name,
            "model_state_dict": model.state_dict(),
            "episode": episode,
            "state_size": model.state_size,
            "action_size": model.action_size,
            "hidden_size": model.hidden_size,
        },
        output_path,
    )


def compute_gae(transitions: List[PPOTransition]) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.array([transition.reward for transition in transitions], dtype=np.float32)
    values = np.array([transition.value for transition in transitions], dtype=np.float32)
    dones = np.array([transition.done for transition in transitions], dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for index in range(len(transitions) - 1, -1, -1):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + GAMMA * next_value * non_terminal - values[index]
        gae = delta + GAMMA * GAE_LAMBDA * non_terminal * gae
        advantages[index] = gae
        next_value = values[index]

    returns = advantages + values
    return returns, advantages


def evaluate_batch(
    model: PPOActorCritic,
    transitions: List[PPOTransition],
    indices: List[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = []
    entropies = []
    values = []

    for index in indices:
        transition = transitions[index]
        state_tensor = torch.tensor(
            transition.state_vec[None, :],
            dtype=torch.float32,
            device=device,
        )
        action_tensor = torch.tensor(
            transition.action_vecs,
            dtype=torch.float32,
            device=device,
        )
        state_batch = state_tensor.repeat(action_tensor.shape[0], 1)

        logits = model.action_logits(state_batch, action_tensor)
        distribution = Categorical(logits=logits)
        action_index = torch.tensor(transition.action_index, dtype=torch.long, device=device)

        log_probs.append(distribution.log_prob(action_index))
        entropies.append(distribution.entropy())
        values.append(model.value(state_tensor).squeeze(0))

    return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def update_policy(
    model: PPOActorCritic,
    optimizer: optim.Optimizer,
    transitions: List[PPOTransition],
    device: str,
    rng: random.Random,
) -> Dict[str, float]:
    if not transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    returns_np, advantages_np = compute_gae(transitions)
    advantages = torch.tensor(advantages_np, dtype=torch.float32, device=device)
    returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    model.train()
    indices = list(range(len(transitions)))
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

    for _ in range(PPO_EPOCHS):
        rng.shuffle(indices)
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            new_log_probs, entropy, values = evaluate_batch(model, transitions, batch_indices, device)

            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped = ratios * batch_advantages
            clipped = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, batch_returns)
            entropy_mean = entropy.mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))

    model.eval()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


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
    ranked = sorted(
        (
            (name, competitor_stats.championship_score, competitor_by_name[name].path)
            for name, competitor_stats in stats.items()
            if name in competitor_by_name
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0] if ranked else None


def setup_trade_curriculum(env: BancoImobiliarioEnv, rng: random.Random):
    groups = [group for group, props in GROUPS.items() if len(props) >= 3]
    if len(groups) < 2:
        return

    group_a, group_b = rng.sample(groups, 2)
    props_a = list(GROUPS[group_a])
    props_b = list(GROUPS[group_b])
    rng.shuffle(props_a)
    rng.shuffle(props_b)

    player_a = rng.randrange(env.num_players)
    player_b = (player_a + rng.randrange(1, env.num_players)) % env.num_players

    for space in env.board:
        if space.is_ownable():
            space.owner = None
            space.houses = 0
            space.mortgaged = False

    env.board[props_a[0]].owner = player_a
    env.board[props_a[1]].owner = player_a
    env.board[props_a[2]].owner = player_b

    env.board[props_b[0]].owner = player_b
    env.board[props_b[1]].owner = player_b
    env.board[props_b[2]].owner = player_a

    available_props = [
        index
        for index, space in enumerate(env.board)
        if space.is_ownable() and space.owner is None
    ]
    rng.shuffle(available_props)
    for prop_index in available_props[: rng.randint(2, 6)]:
        env.board[prop_index].owner = rng.randrange(env.num_players)

    for player in env.players:
        player.money = rng.randint(450, 1500)
        player.position = 0
        player.bankrupt = False
        player.in_jail = False
        player.jail_turns = 0

    env.current_player_index = rng.choice([player_a, player_b])
    env.phase = "ready_to_roll"
    env.trade_draft = None
    env.pending_trade = None
    env.last_trade_result = None
    env.auction = None
    env.trade_proposed_this_turn = False
    env.last_message = "Curriculum de trocas iniciado."


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
    update_every_episodes: int,
    trade_curriculum_ratio: float,
    encoder_name: str,
    hidden_size: int,
):
    encoder_name = normalize_encoder_name(encoder_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Treinando PPO em: {device}")
    print(
        f"Encoder: {encoder_name} | "
        f"state_size={get_state_size(encoder_name)} | "
        f"action_size={get_action_size(encoder_name)} | "
        f"hidden_size={hidden_size}"
    )

    model = PPOActorCritic(
        state_size=get_state_size(encoder_name),
        action_size=get_action_size(encoder_name),
        hidden_size=hidden_size,
    ).to(device)
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = random.Random(seed)
    best_championship_score = float("-inf")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    rollout_buffer: List[PPOTransition] = []
    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    for episode in range(1, episodes + 1):
        env = BancoImobiliarioEnv(num_players=4, seed=seed + episode)
        curriculum_episode = rng.random() < trade_curriculum_ratio
        if curriculum_episode:
            setup_trade_curriculum(env, rng)

        agent = PPOAgent(
            player_id=0,
            model=model,
            device=device,
            deterministic=False,
            encoder=encoder_name,
        )

        transitions: List[PPOTransition] = []
        total_reward = 0.0
        steps = 0
        trade_proposals = 0
        trade_accepts = 0
        trade_declines = 0

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

            selection = agent.select_action(state, valid_actions, deterministic=False)
            action_type = selection.action.get("type")
            _, raw_reward, done, _ = env.step(selection.action)
            reward = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, raw_reward))

            if action_type in ("propose_trade", "submit_trade"):
                trade_proposals += 1
            elif action_type == "accept_trade":
                trade_accepts += 1
            elif action_type == "decline_trade":
                trade_declines += 1

            transitions.append(
                PPOTransition(
                    state_vec=selection.state_vec,
                    action_vecs=selection.action_vecs,
                    action_index=selection.action_index,
                    old_log_prob=selection.log_prob,
                    value=selection.value,
                    reward=reward,
                    done=done,
                )
            )
            total_reward += reward
            steps += 1

        rollout_buffer.extend(transitions)

        should_update = (
            episode % update_every_episodes == 0
            or episode == episodes
        )
        updated = False
        if should_update:
            last_metrics = update_policy(model, optimizer, rollout_buffer, device, rng)
            rollout_buffer = []
            updated = True

        should_save_main = episode % SAVE_EVERY == 0 or episode == episodes
        should_save_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_every > 0
            and (episode % checkpoint_every == 0 or episode == episodes)
        )

        if should_save_main:
            save_checkpoint(model, episode, output_path, encoder_name)

        if should_save_checkpoint:
            checkpoint_prefix = "ppo_raw_ep" if encoder_name == ENCODER_RAW else "ppo_ep"
            checkpoint_path = str(Path(checkpoint_dir) / f"{checkpoint_prefix}_{episode:06d}.pt")
            save_checkpoint(model, episode, checkpoint_path, encoder_name)

        if (
            checkpoint_dir is not None
            and tournament_every > 0
            and tournament_games > 0
            and (episode % tournament_every == 0 or episode == episodes)
        ):
            print("")
            print(f"=== Campeonato de checkpoints PPO no episodio {episode:04d} ===")
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
                        f"Novo melhor checkpoint PPO: {best_name} "
                        f"(score={best_score:.2f}) salvo em {best_output_path}"
                    )
            print("=== Fim do campeonato ===")
            print("")

        winner = env.winner
        winner_name = env.players[winner].name if winner is not None else "nenhum"
        update_marker = "sim" if updated else "nao"
        curriculum_marker = "sim" if curriculum_episode else "nao"
        print(
            f"Episodio {episode:04d} | "
            f"steps={steps:04d} | "
            f"reward={total_reward:8.2f} | "
            f"update={update_marker} | "
            f"trade_curr={curriculum_marker} | "
            f"trades={trade_proposals}/{trade_accepts}/{trade_declines} | "
            f"loss={last_metrics['loss']:.5f} | "
            f"policy={last_metrics['policy_loss']:.5f} | "
            f"value={last_metrics['value_loss']:.5f} | "
            f"entropy={last_metrics['entropy']:.5f} | "
            f"vencedor={winner_name}"
        )

    print(f"Modelo PPO salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, choices=ENCODER_NAMES, default=ENCODER_RAW)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--tournament-every", type=int, default=0)
    parser.add_argument("--tournament-games", type=int, default=24)
    parser.add_argument("--tournament-latest", type=int, default=8)
    parser.add_argument("--best-output", type=str, default=None)
    parser.add_argument("--update-every-episodes", type=int, default=UPDATE_EVERY_EPISODES)
    parser.add_argument("--trade-curriculum-ratio", type=float, default=TRADE_CURRICULUM_RATIO)
    args = parser.parse_args()

    encoder_name = normalize_encoder_name(args.encoder)
    if encoder_name == ENCODER_RAW:
        output_path = args.output or "models/ppo_raw_agent.pt"
        checkpoint_dir = args.checkpoint_dir or "models/ppo_raw_checkpoints"
        best_output_path = args.best_output or "models/best_ppo_raw_agent.pt"
    else:
        output_path = args.output or "models/ppo_agent.pt"
        checkpoint_dir = args.checkpoint_dir or "models/ppo_checkpoints"
        best_output_path = args.best_output or "models/best_ppo_agent.pt"

    train(
        episodes=args.episodes,
        seed=args.seed,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        tournament_every=args.tournament_every,
        tournament_games=args.tournament_games,
        tournament_latest=args.tournament_latest,
        best_output_path=best_output_path,
        update_every_episodes=max(1, args.update_every_episodes),
        trade_curriculum_ratio=max(0.0, min(1.0, args.trade_curriculum_ratio)),
        encoder_name=encoder_name,
        hidden_size=max(32, args.hidden_size),
    )


if __name__ == "__main__":
    main()
