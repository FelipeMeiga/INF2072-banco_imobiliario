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
ENTROPY_COEF = 0.05
MAX_GRAD_NORM = 0.50
SAVE_EVERY = 50
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0
UPDATE_EVERY_EPISODES = 4
TRADE_CURRICULUM_RATIO = 0.50


@dataclass
class PPOTransition:
    state_vec: np.ndarray
    action_vecs: np.ndarray
    action_index: int
    player_id: int
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
    optimizer: optim.Optimizer,
    episode: int,
    output_path: str,
    encoder_name: str,
    rng: random.Random,
    best_championship_score: float,
    rollout_buffer: List[PPOTransition],
    training_config: Dict[str, object],
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "algorithm": "ppo",
            "encoder": encoder_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "state_size": model.state_size,
            "action_size": model.action_size,
            "hidden_size": model.hidden_size,
            "best_championship_score": best_championship_score,
            "rollout_buffer": serialize_rollout_buffer(rollout_buffer),
            "python_random_state": random.getstate(),
            "numpy_random_state": serialize_numpy_random_state(np.random.get_state()),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_rng_state": rng.getstate(),
            "training_config": training_config,
        },
        output_path,
    )


def load_torch_checkpoint(path: str, device: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def serialize_numpy_random_state(state: tuple) -> Dict[str, object]:
    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def deserialize_numpy_random_state(state: object) -> tuple:
    if isinstance(state, dict):
        return (
            state["name"],
            np.array(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    return state


def serialize_rollout_buffer(transitions: List[PPOTransition]) -> List[Dict[str, object]]:
    return [
        {
            "state_vec": transition.state_vec.tolist(),
            "action_vecs": transition.action_vecs.tolist(),
            "action_index": int(transition.action_index),
            "player_id": int(transition.player_id),
            "old_log_prob": float(transition.old_log_prob),
            "value": float(transition.value),
            "reward": float(transition.reward),
            "done": bool(transition.done),
        }
        for transition in transitions
    ]


def deserialize_rollout_buffer(raw_transitions: object) -> List[PPOTransition]:
    if not raw_transitions:
        return []

    transitions = []
    for item in raw_transitions:
        transitions.append(
            PPOTransition(
                state_vec=np.array(item["state_vec"], dtype=np.float32),
                action_vecs=np.array(item["action_vecs"], dtype=np.float32),
                action_index=int(item["action_index"]),
                player_id=int(item.get("player_id", 0)),
                old_log_prob=float(item["old_log_prob"]),
                value=float(item["value"]),
                reward=float(item["reward"]),
                done=bool(item["done"]),
            )
        )
    return transitions


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


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
    entropy_coef: float,
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
            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy_mean

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

    preferred_pairs = [
        ("Orange", "Pink"),
        ("Orange", "Red"),
        ("Red", "Yellow"),
        ("Yellow", "Green"),
        ("Light Blue", "Pink"),
    ]
    valid_preferred_pairs = [
        pair
        for pair in preferred_pairs
        if pair[0] in GROUPS and pair[1] in GROUPS
    ]
    if valid_preferred_pairs and rng.random() < 0.65:
        group_a, group_b = rng.choice(valid_preferred_pairs)
        if rng.random() < 0.5:
            group_a, group_b = group_b, group_a
    else:
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
        player.money = rng.randint(700, 1800)
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
    resume_path: Optional[str],
    entropy_coef: float,
):
    encoder_name = normalize_encoder_name(encoder_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resume_checkpoint: Optional[Dict[str, object]] = None
    resume_episode = 0
    if resume_path:
        resume_checkpoint = load_torch_checkpoint(resume_path, device)
        if resume_checkpoint.get("algorithm") != "ppo":
            raise ValueError(f"Checkpoint nao e PPO: {resume_path}")

        checkpoint_encoder = normalize_encoder_name(resume_checkpoint.get("encoder", encoder_name))
        if checkpoint_encoder != encoder_name:
            raise ValueError(
                f"Checkpoint usa encoder '{checkpoint_encoder}', mas o treino pediu "
                f"'{encoder_name}'. Use --encoder {checkpoint_encoder}."
            )

        expected_state_size = get_state_size(encoder_name)
        expected_action_size = get_action_size(encoder_name)
        state_size = int(resume_checkpoint.get("state_size", expected_state_size))
        action_size = int(resume_checkpoint.get("action_size", expected_action_size))
        if state_size != expected_state_size or action_size != expected_action_size:
            raise ValueError(
                "Checkpoint PPO incompativel com o estado/acao atual: "
                f"salvo=({state_size}, {action_size}), "
                f"esperado=({expected_state_size}, {expected_action_size})"
            )

        checkpoint_hidden_size = int(resume_checkpoint.get("hidden_size", hidden_size))
        if checkpoint_hidden_size != hidden_size:
            print(
                f"Usando hidden_size={checkpoint_hidden_size} do checkpoint "
                f"(valor recebido: {hidden_size})."
            )
            hidden_size = checkpoint_hidden_size

        training_config = resume_checkpoint.get("training_config")
        if isinstance(training_config, dict) and "seed" in training_config:
            checkpoint_seed = int(training_config["seed"])
            if checkpoint_seed != seed:
                print(
                    f"Usando seed={checkpoint_seed} do checkpoint "
                    f"(valor recebido: {seed})."
                )
                seed = checkpoint_seed

        resume_episode = int(resume_checkpoint.get("episode", 0))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Treinando PPO em: {device}")
    print(
        f"Encoder: {encoder_name} | "
        f"state_size={get_state_size(encoder_name)} | "
        f"action_size={get_action_size(encoder_name)} | "
        f"hidden_size={hidden_size} | "
        f"entropy_coef={entropy_coef}"
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
    start_episode = 1
    rollout_buffer: List[PPOTransition] = []

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, device)
        else:
            print("Checkpoint antigo sem optimizer_state_dict; otimizador reiniciado.")

        if "python_random_state" in resume_checkpoint:
            random.setstate(resume_checkpoint["python_random_state"])
        else:
            print("Checkpoint antigo sem estado do random global; usando seed informado.")

        if "numpy_random_state" in resume_checkpoint:
            np.random.set_state(deserialize_numpy_random_state(resume_checkpoint["numpy_random_state"]))
        else:
            print("Checkpoint antigo sem estado do NumPy; usando seed informado.")

        if "torch_random_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_random_state"].cpu())
        else:
            print("Checkpoint antigo sem estado do Torch; usando seed informado.")

        cuda_rng_state = resume_checkpoint.get("torch_cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

        if "train_rng_state" in resume_checkpoint:
            rng.setstate(resume_checkpoint["train_rng_state"])
        else:
            print("Checkpoint antigo sem estado do RNG local; usando seed informado.")

        best_championship_score = float(
            resume_checkpoint.get("best_championship_score", best_championship_score)
        )
        rollout_buffer = deserialize_rollout_buffer(resume_checkpoint.get("rollout_buffer", []))
        start_episode = resume_episode + 1
        print(
            f"Retomando PPO de {resume_path} no episodio {resume_episode}. "
            f"Proximo episodio: {start_episode}."
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    training_config = {
        "seed": seed,
        "encoder": encoder_name,
        "hidden_size": hidden_size,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "ppo_epochs": PPO_EPOCHS,
        "clip_epsilon": CLIP_EPSILON,
        "value_coef": VALUE_COEF,
        "entropy_coef": entropy_coef,
        "max_grad_norm": MAX_GRAD_NORM,
        "reward_clip_min": REWARD_CLIP_MIN,
        "reward_clip_max": REWARD_CLIP_MAX,
        "update_every_episodes": update_every_episodes,
        "trade_curriculum_ratio": trade_curriculum_ratio,
    }

    if start_episode > episodes:
        print(
            f"Checkpoint ja esta no episodio {resume_episode}; "
            f"nada a treinar ate --episodes {episodes}."
        )
        return

    for episode in range(start_episode, episodes + 1):
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

        transitions_by_player: List[List[PPOTransition]] = [
            []
            for _ in range(env.num_players)
        ]
        total_reward = 0.0
        steps = 0
        trade_starts = 0
        trade_submits = 0
        trade_cancels = 0
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
            _, raw_reward, done, info = env.step(selection.action)
            rewards_by_player = info.get("rewards_by_player")
            if not isinstance(rewards_by_player, list):
                rewards_by_player = [0.0 for _ in range(env.num_players)]
                rewards_by_player[acting_player] = raw_reward
            clipped_rewards_by_player = [
                max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, float(reward)))
                for reward in rewards_by_player[: env.num_players]
            ]
            while len(clipped_rewards_by_player) < env.num_players:
                clipped_rewards_by_player.append(0.0)

            if action_type == "start_trade":
                trade_starts += 1
            elif action_type in ("propose_trade", "submit_trade"):
                trade_submits += 1
            elif action_type == "cancel_trade":
                trade_cancels += 1
            elif action_type == "accept_trade":
                trade_accepts += 1
            elif action_type == "decline_trade":
                trade_declines += 1

            transition = PPOTransition(
                state_vec=selection.state_vec,
                action_vecs=selection.action_vecs,
                action_index=selection.action_index,
                player_id=acting_player,
                old_log_prob=selection.log_prob,
                value=selection.value,
                reward=clipped_rewards_by_player[acting_player],
                done=done,
            )
            transitions_by_player[acting_player].append(transition)

            for player_index, player_reward in enumerate(clipped_rewards_by_player):
                if player_index == acting_player or not transitions_by_player[player_index]:
                    continue
                transitions_by_player[player_index][-1].reward += player_reward

            total_reward += clipped_rewards_by_player[acting_player]
            steps += 1

        for player_transitions in transitions_by_player:
            if player_transitions:
                player_transitions[-1].done = True
                rollout_buffer.extend(player_transitions)

        should_update = (
            episode % update_every_episodes == 0
            or episode == episodes
        )
        updated = False
        if should_update:
            last_metrics = update_policy(
                model,
                optimizer,
                rollout_buffer,
                device,
                rng,
                entropy_coef=entropy_coef,
            )
            rollout_buffer = []
            updated = True

        should_save_main = episode % SAVE_EVERY == 0 or episode == episodes
        should_save_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_every > 0
            and (episode % checkpoint_every == 0 or episode == episodes)
        )
        checkpoint_path: Optional[str] = None

        if should_save_main:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                output_path=output_path,
                encoder_name=encoder_name,
                rng=rng,
                best_championship_score=best_championship_score,
                rollout_buffer=rollout_buffer,
                training_config=training_config,
            )

        if should_save_checkpoint:
            checkpoint_prefix = "ppo_raw_ep" if encoder_name == ENCODER_RAW else "ppo_ep"
            checkpoint_path = str(Path(checkpoint_dir) / f"{checkpoint_prefix}_{episode:06d}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                output_path=checkpoint_path,
                encoder_name=encoder_name,
                rng=rng,
                best_championship_score=best_championship_score,
                rollout_buffer=rollout_buffer,
                training_config=training_config,
            )

        if (
            checkpoint_dir is not None
            and tournament_every > 0
            and tournament_games > 0
            and (episode % tournament_every == 0 or episode == episodes)
        ):
            best_score_updated = False
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
                    best_score_updated = True
                    best_dir = os.path.dirname(best_output_path)
                    if best_dir:
                        os.makedirs(best_dir, exist_ok=True)
                    shutil.copyfile(best_path, best_output_path)
                    print(
                        f"Novo melhor checkpoint PPO: {best_name} "
                        f"(score={best_score:.2f}) salvo em {best_output_path}"
                    )
            if best_score_updated:
                if should_save_main:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        episode=episode,
                        output_path=output_path,
                        encoder_name=encoder_name,
                        rng=rng,
                        best_championship_score=best_championship_score,
                        rollout_buffer=rollout_buffer,
                        training_config=training_config,
                    )
                if should_save_checkpoint and checkpoint_path is not None:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        episode=episode,
                        output_path=checkpoint_path,
                        encoder_name=encoder_name,
                        rng=rng,
                        best_championship_score=best_championship_score,
                        rollout_buffer=rollout_buffer,
                        training_config=training_config,
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
            f"trades={trade_starts}/{trade_submits}/{trade_cancels}/{trade_accepts}/{trade_declines} | "
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--update-every-episodes", type=int, default=UPDATE_EVERY_EPISODES)
    parser.add_argument("--trade-curriculum-ratio", type=float, default=TRADE_CURRICULUM_RATIO)
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
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
        resume_path=args.resume,
        entropy_coef=max(0.0, args.entropy_coef),
    )


if __name__ == "__main__":
    main()
