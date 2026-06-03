from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from game.actions import (
    ACCEPT_TRADE,
    AUCTION_BID,
    AUCTION_PASS,
    BUY_PROPERTY,
    DECLINE_TRADE,
    MORTGAGE_PROPERTY,
    PASS_BUY,
    PAY_JAIL_FINE,
    PROPOSE_TRADE,
    ROLL_DICE,
    BUILD_HOUSE,
    SELL_HOUSE,
    UNMORTGAGE_PROPERTY,
    USE_JAIL_CARD,
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
    SELL_HOUSE,
    MORTGAGE_PROPERTY,
    UNMORTGAGE_PROPERTY,
    PAY_JAIL_FINE,
    USE_JAIL_CARD,
    AUCTION_BID,
    AUCTION_PASS,
]

PHASES = [
    "ready_to_roll",
    "awaiting_buy",
    "pending_trade_response",
    "auction",
    "game_over",
]

STATE_SIZE = (
    4
    + len(PHASES)
    + 2
    + 1
    + 4
    + 4
    + 4
    + 4
    + 4
    + 4
    + (4 * 6)
    + (NUM_SPACES * 6)
    + NUM_SPACES
    + NUM_SPACES
    + 1
    + 1
    + 1
)
ACTION_SIZE = len(ACTION_TYPES) + 4 + NUM_SPACES + NUM_SPACES + NUM_SPACES + 3


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
    values.append(float(state.get("consecutive_doubles", 0)) / 3.0)

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

    in_jail = list(state.get("player_in_jail", []))[:4]
    while len(in_jail) < 4:
        in_jail.append(False)
    values.extend([1.0 if b else 0.0 for b in in_jail])

    jail_turns = list(state.get("player_jail_turns", []))[:4]
    while len(jail_turns) < 4:
        jail_turns.append(0)
    values.extend([float(t) / 3.0 for t in jail_turns])

    jail_cards = list(state.get("player_jail_free_cards", []))[:4]
    while len(jail_cards) < 4:
        jail_cards.append(0)
    values.extend([float(c) / 2.0 for c in jail_cards])

    net_worth = list(state.get("player_net_worth", []))[:4]
    while len(net_worth) < 4:
        net_worth.append(0)
    values.extend([max(-1.0, min(3.0, float(v) / 3000.0)) for v in net_worth])

    completed_groups = list(state.get("player_completed_groups", []))[:4]
    while len(completed_groups) < 4:
        completed_groups.append(0)
    values.extend([float(v) / 8.0 for v in completed_groups])

    near_groups = list(state.get("player_near_groups", []))[:4]
    while len(near_groups) < 4:
        near_groups.append(0)
    values.extend([float(v) / 8.0 for v in near_groups])

    blocking_properties = list(state.get("player_blocking_properties", []))[:4]
    while len(blocking_properties) < 4:
        blocking_properties.append(0)
    values.extend([float(v) / 8.0 for v in blocking_properties])

    highest_rent = list(state.get("player_highest_rent_potential", []))[:4]
    while len(highest_rent) < 4:
        highest_rent.append(0)
    values.extend([float(v) / 2000.0 for v in highest_rent])

    group_progress = list(state.get("player_group_progress", []))[:4]
    while len(group_progress) < 4:
        group_progress.append(0)
    values.extend([float(v) for v in group_progress])

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

    mortgaged = list(state.get("property_mortgaged", []))[:NUM_SPACES]
    while len(mortgaged) < NUM_SPACES:
        mortgaged.append(False)
    values.extend([1.0 if m else 0.0 for m in mortgaged])

    values.append(float(state.get("turn_count", 0)) / 2000.0)
    values.append(float(state.get("bank_houses", 32)) / 32.0)
    values.append(float(state.get("bank_hotels", 12)) / 12.0)

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

    property_index = int(action.get("property_index", -1))
    values.extend(_one_hot(property_index, NUM_SPACES))

    values.append(float(action.get("offer_money", 0)) / 1500.0)
    values.append(float(action.get("request_money", 0)) / 1500.0)
    values.append(float(action.get("amount", 0)) / 1500.0)

    return np.array(values, dtype=np.float32)
