from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from game.encoders import ACTION_SIZE, STATE_SIZE, encode_action, encode_state


class PPOActorCritic(nn.Module):
    """
    Actor-critic for variable action sets.

    The actor scores each valid (state, action) pair. The softmax is applied only
    over the valid actions supplied by the environment, which gives us action
    masking without requiring a fixed action head.
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(STATE_SIZE + ACTION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def action_logits(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state_batch, action_batch], dim=1)
        return self.actor(state_action).squeeze(-1)

    def value(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.critic(state_batch).squeeze(-1)


@dataclass
class PPOActionSelection:
    action: Dict[str, Any]
    action_index: int
    log_prob: float
    value: float
    state_vec: np.ndarray
    action_vecs: np.ndarray


class PPOAgent:
    def __init__(
        self,
        player_id: int,
        model: Optional[PPOActorCritic] = None,
        device: Optional[str] = None,
        deterministic: bool = True,
    ):
        self.player_id = player_id
        self.model = model or PPOActorCritic()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.deterministic = deterministic
        self.model.to(self.device)
        self.model.eval()

    def select_action(
        self,
        state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
        deterministic: Optional[bool] = None,
    ) -> PPOActionSelection:
        if not valid_actions:
            state_vec = encode_state(state)
            return PPOActionSelection(
                action={"type": "no_action"},
                action_index=0,
                log_prob=0.0,
                value=0.0,
                state_vec=state_vec,
                action_vecs=np.zeros((1, ACTION_SIZE), dtype=np.float32),
            )

        deterministic = self.deterministic if deterministic is None else deterministic
        state_vec = encode_state(state)
        action_vecs = np.stack([encode_action(action) for action in valid_actions], axis=0)

        state_batch = np.repeat(state_vec[None, :], len(valid_actions), axis=0)
        state_tensor = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_vecs, dtype=torch.float32, device=self.device)
        value_tensor = torch.tensor(state_vec[None, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model.action_logits(state_tensor, action_tensor)
            value = float(self.model.value(value_tensor).item())
            if deterministic:
                action_index = int(torch.argmax(logits).item())
                log_probs = torch.log_softmax(logits, dim=0)
                log_prob = float(log_probs[action_index].item())
            else:
                distribution = Categorical(logits=logits)
                sampled = distribution.sample()
                action_index = int(sampled.item())
                log_prob = float(distribution.log_prob(sampled).item())

        return PPOActionSelection(
            action=valid_actions[action_index],
            action_index=action_index,
            log_prob=log_prob,
            value=value,
            state_vec=state_vec,
            action_vecs=action_vecs,
        )

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.select_action(state, valid_actions).action

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
