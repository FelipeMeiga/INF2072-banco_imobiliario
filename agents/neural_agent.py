from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from game.actions import ACCEPT_TRADE, BUILD_HOUSE, BUY_PROPERTY, PROPOSE_TRADE
from game.encoders import ACTION_SIZE, STATE_SIZE, encode_action, encode_state


class QNetwork(nn.Module):
    """
    Rede que recebe o par (estado, ação) e devolve um valor Q.

    Em vez de a rede ter uma saída fixa para cada ação possível, ela avalia uma ação
    estruturada específica. Isso combina melhor com trocas, porque uma troca carrega
    parâmetros como propriedades e dinheiro.
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE + ACTION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_action).squeeze(-1)


class NeuralAgent:
    """
    Agente neural baseado em Q-learning/DQN.

    O agente escolhe a ação válida com maior Q estimado. Durante treinamento, usa
    epsilon-greedy para explorar ações aleatórias.
    """

    def __init__(
        self,
        player_id: int,
        model: Optional[QNetwork] = None,
        epsilon: float = 0.05,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        buy_probability: float = 1.0,
        build_probability: float = 0.95,
        trade_probability: float = 0.45,
        accept_trade_probability: float = 0.75,
    ):
        self.player_id = player_id
        self.model = model or QNetwork()
        self.epsilon = epsilon
        self.buy_probability = buy_probability
        self.build_probability = build_probability
        self.trade_probability = trade_probability
        self.accept_trade_probability = accept_trade_probability
        self.random = random.Random(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        strategic_action = self._choose_strategic_action(state, valid_actions)
        if strategic_action is not None:
            return strategic_action

        if self.random.random() < self.epsilon:
            return self.random.choice(valid_actions)

        state_vec = encode_state(state)
        state_batch = np.repeat(state_vec[None, :], len(valid_actions), axis=0)
        action_batch = np.stack([encode_action(action) for action in valid_actions], axis=0)
        state_action_batch = np.concatenate([state_batch, action_batch], axis=1)

        with torch.no_grad():
            tensor = torch.tensor(state_action_batch, dtype=torch.float32, device=self.device)
            q_values = self.model(tensor).detach().cpu().numpy()

        best_index = int(np.argmax(q_values))
        return valid_actions[best_index]

    def _choose_strategic_action(
        self,
        state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        buy_actions = [a for a in valid_actions if a["type"] == BUY_PROPERTY]
        if buy_actions and self.random.random() < self.buy_probability:
            return buy_actions[0]

        build_actions = [a for a in valid_actions if a["type"] == BUILD_HOUSE]
        if build_actions and self.random.random() < self.build_probability:
            return self._choose_build_action(state, build_actions)

        accept_actions = [a for a in valid_actions if a["type"] == ACCEPT_TRADE]
        if accept_actions and self.random.random() < self.accept_trade_probability:
            return accept_actions[0]

        trade_actions = [a for a in valid_actions if a["type"] == PROPOSE_TRADE]
        if trade_actions and self.random.random() < self.trade_probability:
            return self._choose_trade_action(state, trade_actions)

        return None

    def _choose_build_action(
        self,
        state: Dict[str, Any],
        build_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        houses = state.get("property_houses", [])

        def build_score(action: Dict[str, Any]) -> tuple[int, int]:
            prop_index = int(action.get("property_index", -1))
            house_count = houses[prop_index] if 0 <= prop_index < len(houses) else 0
            return int(house_count), prop_index

        return max(build_actions, key=build_score)

    def _choose_trade_action(
        self,
        state: Dict[str, Any],
        trade_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        property_owners = state.get("property_owners", [])
        property_groups = state.get("property_groups", [])

        def trade_score(action: Dict[str, Any]) -> tuple[int, int, int, int]:
            request_properties = action.get("request_properties", [])
            offer_properties = action.get("offer_properties", [])
            request_money = int(action.get("request_money", 0))
            offer_money = int(action.get("offer_money", 0))
            completion_bonus = sum(
                self._would_complete_group(prop_index, property_owners, property_groups)
                for prop_index in request_properties
            )
            useful_group_bonus = sum(
                self._owned_group_count(prop_index, property_owners, property_groups)
                for prop_index in request_properties
            )
            offered_group_penalty = sum(
                self._owned_group_count(prop_index, property_owners, property_groups)
                for prop_index in offer_properties
            )

            return (
                completion_bonus,
                useful_group_bonus - offered_group_penalty,
                -request_money,
                offer_money,
            )

        return max(trade_actions, key=trade_score)

    def _would_complete_group(
        self,
        prop_index: int,
        property_owners: List[Any],
        property_groups: List[Any],
    ) -> int:
        if prop_index < 0 or prop_index >= len(property_groups):
            return 0

        group = property_groups[prop_index]
        if group is None:
            return 0

        group_indexes = [
            i
            for i, candidate_group in enumerate(property_groups)
            if candidate_group == group
        ]
        if not group_indexes:
            return 0

        return int(
            all(
                i == prop_index
                or (i < len(property_owners) and property_owners[i] == self.player_id)
                for i in group_indexes
            )
        )

    def _owned_group_count(
        self,
        prop_index: int,
        property_owners: List[Any],
        property_groups: List[Any],
    ) -> int:
        if prop_index < 0 or prop_index >= len(property_groups):
            return 0

        group = property_groups[prop_index]
        if group is None:
            return 0

        return sum(
            1
            for i, candidate_group in enumerate(property_groups)
            if candidate_group == group
            and i < len(property_owners)
            and property_owners[i] == self.player_id
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
