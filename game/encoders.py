from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from game.actions import (
    ACCEPT_TRADE,
    BUY_PROPERTY,
    DECLINE_TRADE,
    PASS_BUY,
    PROPOSE_TRADE,
    ROLL_DICE,
    BUILD_HOUSE,
)
from game.board import NUM_SPACES

ACTION_TYPES = [
    ROLL_DICE,
    BUY_PROPERTY,
    PASS_BUY,
    PROPOSE_TRADE,
    ACCEPT_TRADE,
    DECLINE_TRADE,
    BUILD_HOUSE,
]

PHASES = [
    "ready_to_roll",
    "awaiting_buy",
    "pending_trade_response",
    "game_over",
]

STATE_SIZE = 4 + 4 + 2 + 4 + 4 + 4 + (NUM_SPACES * 6) + NUM_SPACES + 1
ACTION_SIZE = len(ACTION_TYPES) + 4 + NUM_SPACES + NUM_SPACES + NUM_SPACES + 2


def _one_hot(index: int, size: int) -> List[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def encode_state(state: Dict[str, Any], num_players: int = 4) -> np.ndarray:
    """
    Transforma o estado estruturado do ambiente em um vetor numérico fixo.

    Essa versão assume no máximo 4 jogadores, que é o padrão do projeto.
    """

    values: List[float] = []

    current_player = int(state.get("current_player", 0))
    values.extend(_one_hot(current_player, 4))

    phase = state.get("phase", "ready_to_roll")
    phase_index = PHASES.index(phase) if phase in PHASES else 0
    values.extend(_one_hot(phase_index, len(PHASES)))

    dice = state.get("dice", (1, 1))
    values.append(float(dice[0]) / 6.0)
    values.append(float(dice[1]) / 6.0)

    money = list(state.get("player_money", []))[:4]
    while len(money) < 4:
        money.append(0)
    values.extend([float(m) / 3000.0 for m in money])

    positions = list(state.get("player_positions", []))[:4]
    while len(positions) < 4:
        positions.append(0)
    values.extend([float(p) / float(NUM_SPACES - 1) for p in positions])

    bankrupt = list(state.get("player_bankrupt", []))[:4]
    while len(bankrupt) < 4:
        bankrupt.append(True)
    values.extend([1.0 if b else 0.0 for b in bankrupt])

    # Para cada casa, codifica o dono:
    # 0 = não é propriedade
    # 1 = propriedade sem dono
    # 2..5 = dono jogador 0..3
    owners = list(state.get("property_owners", []))[:NUM_SPACES]
    while len(owners) < NUM_SPACES:
        owners.append(-2)

    for owner in owners:
        if owner == -2:
            owner_index = 0
        elif owner is None or owner == -1:
            owner_index = 1
        else:
            owner_index = 2 + int(owner)
        values.extend(_one_hot(owner_index, 6))

    houses = list(state.get("property_houses", []))[:NUM_SPACES]
    while len(houses) < NUM_SPACES:
        houses.append(0)
    values.extend([float(h) / 5.0 for h in houses])

    values.append(float(state.get("turn_count", 0)) / 500.0)

    return np.array(values, dtype=np.float32)


def encode_action(action: Dict[str, Any], num_players: int = 4) -> np.ndarray:
    """
    Transforma uma ação estruturada em vetor.

    Isso permite que a rede estime Q(estado, ação), inclusive para ações com parâmetros,
    como propostas de troca.
    """

    values: List[float] = []

    action_type = action.get("type")
    action_index = ACTION_TYPES.index(action_type) if action_type in ACTION_TYPES else -1
    values.extend(_one_hot(action_index, len(ACTION_TYPES)))

    target_player = action.get("target_player", -1)
    values.extend(_one_hot(int(target_player), 4))

    offer_properties = set(action.get("offer_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in offer_properties else 0.0)

    request_properties = set(action.get("request_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in request_properties else 0.0)

    build_property = int(action.get("property_index", -1))
    values.extend(_one_hot(build_property, NUM_SPACES))

    values.append(float(action.get("offer_money", 0)) / 1500.0)
    values.append(float(action.get("request_money", 0)) / 1500.0)

    return np.array(values, dtype=np.float32)
