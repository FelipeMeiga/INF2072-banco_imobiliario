from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from game.actions import (
    ACCEPT_TRADE,
    ADD_TRADE_OFFER_PROPERTY,
    ADD_TRADE_REQUEST_PROPERTY,
    AUCTION_BID,
    AUCTION_PASS,
    BUY_PROPERTY,
    CANCEL_TRADE,
    DECLINE_TRADE,
    FINISH_TRADE_OFFER,
    FINISH_TRADE_REQUEST,
    MORTGAGE_PROPERTY,
    PASS_BUY,
    PAY_JAIL_FINE,
    PROPOSE_TRADE,
    ROLL_DICE,
    BUILD_HOUSE,
    SELL_HOUSE,
    SELECT_TRADE_TARGET,
    SET_TRADE_OFFER_MONEY,
    SET_TRADE_REQUEST_MONEY,
    START_TRADE,
    SUBMIT_TRADE,
    UNMORTGAGE_PROPERTY,
    USE_JAIL_CARD,
)
from game.board import NUM_SPACES

TRADE_FEATURE_SIZE = 12
TRADE_STAGES = [
    "none",
    "target",
    "offer_properties",
    "offer_money",
    "request_properties",
    "request_money",
    "confirm",
    "response",
]
TRADE_CONTEXT_SIZE = 4 + 4 + len(TRADE_STAGES) + NUM_SPACES + NUM_SPACES + 2
ENCODER_RICH = "rich"
ENCODER_RAW = "raw"
DEFAULT_ENCODER = ENCODER_RICH
ENCODER_NAMES = (ENCODER_RICH, ENCODER_RAW)

ACTION_TYPES = [
    ROLL_DICE,
    BUY_PROPERTY,
    PASS_BUY,
    START_TRADE,
    SELECT_TRADE_TARGET,
    ADD_TRADE_OFFER_PROPERTY,
    FINISH_TRADE_OFFER,
    SET_TRADE_OFFER_MONEY,
    ADD_TRADE_REQUEST_PROPERTY,
    FINISH_TRADE_REQUEST,
    SET_TRADE_REQUEST_MONEY,
    SUBMIT_TRADE,
    CANCEL_TRADE,
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
    "building_trade",
    "pending_trade_response",
    "auction",
    "game_over",
]

RICH_STATE_SIZE = (
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
    + TRADE_CONTEXT_SIZE
    + 1
    + 1
    + 1
)
RICH_ACTION_SIZE = len(ACTION_TYPES) + 4 + NUM_SPACES + NUM_SPACES + NUM_SPACES + 3 + TRADE_FEATURE_SIZE

RAW_STATE_SIZE = (
    4
    + len(PHASES)
    + 5
    + (4 * 6)
    + NUM_SPACES
    + NUM_SPACES
    + NUM_SPACES
    + NUM_SPACES
    + 1
    + 4
    + TRADE_CONTEXT_SIZE
    + 4
)
RAW_ACTION_SIZE = len(ACTION_TYPES) + 4 + NUM_SPACES + NUM_SPACES + NUM_SPACES + 3

STATE_SIZE = RICH_STATE_SIZE
ACTION_SIZE = RICH_ACTION_SIZE


def normalize_encoder_name(encoder: str | None) -> str:
    if encoder is None:
        return DEFAULT_ENCODER
    normalized = encoder.lower().strip()
    if normalized not in ENCODER_NAMES:
        raise ValueError(f"Encoder invalido: {encoder}. Use: {', '.join(ENCODER_NAMES)}")
    return normalized


def get_state_size(encoder: str | None = DEFAULT_ENCODER) -> int:
    return RAW_STATE_SIZE if normalize_encoder_name(encoder) == ENCODER_RAW else RICH_STATE_SIZE


def get_action_size(encoder: str | None = DEFAULT_ENCODER) -> int:
    return RAW_ACTION_SIZE if normalize_encoder_name(encoder) == ENCODER_RAW else RICH_ACTION_SIZE


def _one_hot(index: int, size: int) -> List[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _int_or_default(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _trade_context_features(state: Dict[str, Any]) -> List[float]:
    trade = state.get("trade_draft")
    stage = "none"

    if isinstance(trade, dict):
        stage = str(trade.get("stage", "target"))
    else:
        trade = state.get("pending_trade")
        if isinstance(trade, dict):
            stage = "response"
        else:
            trade = {}

    if stage not in TRADE_STAGES:
        stage = "none"

    from_index = _int_or_default(trade.get("from", -1))
    to_index = _int_or_default(trade.get("to", -1))
    offer_properties = set(trade.get("offer_properties", []))
    request_properties = set(trade.get("request_properties", []))

    values: List[float] = []
    values.extend(_one_hot(from_index, 4))
    values.extend(_one_hot(to_index, 4))
    values.extend(_one_hot(TRADE_STAGES.index(stage), len(TRADE_STAGES)))

    for i in range(NUM_SPACES):
        values.append(1.0 if i in offer_properties else 0.0)

    for i in range(NUM_SPACES):
        values.append(1.0 if i in request_properties else 0.0)

    values.append(float(trade.get("offer_money", 0)) / 1500.0)
    values.append(float(trade.get("request_money", 0)) / 1500.0)
    return values


def _trade_features(action: Dict[str, Any]) -> List[float]:
    raw_features = action.get("trade_features", [])
    if isinstance(raw_features, dict):
        ordered = [
            raw_features.get("proposer_delta", 0.0),
            raw_features.get("target_delta", 0.0),
            raw_features.get("delta_diff", 0.0),
            raw_features.get("offer_value", 0.0),
            raw_features.get("request_value", 0.0),
            raw_features.get("offer_money", 0.0),
            raw_features.get("request_money", 0.0),
            raw_features.get("offer_completes_target", 0.0),
            raw_features.get("request_completes_proposer", 0.0),
            raw_features.get("offer_near_target", 0.0),
            raw_features.get("request_near_proposer", 0.0),
            raw_features.get("offer_critical_proposer", 0.0),
        ]
    else:
        ordered = list(raw_features)

    while len(ordered) < TRADE_FEATURE_SIZE:
        ordered.append(0.0)

    return [
        _clamp(float(value), -3.0, 3.0)
        for value in ordered[:TRADE_FEATURE_SIZE]
    ]


def encode_state(
    state: Dict[str, Any],
    num_players: int = 4,
    encoder: str | None = DEFAULT_ENCODER,
) -> np.ndarray:
    encoder = normalize_encoder_name(encoder)
    if encoder == ENCODER_RAW:
        return encode_state_raw(state, num_players=num_players)
    return encode_state_rich(state, num_players=num_players)


def encode_state_rich(state: Dict[str, Any], num_players: int = 4) -> np.ndarray:
    """
    Transforma o estado estruturado do ambiente em um vetor numérico fixo.

    Essa versão assume no máximo 4 jogadores, que é o padrão do projeto.
    """

    values: List[float] = []

    current_player = _int_or_default(state.get("current_player", 0), 0)
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

    values.extend(_trade_context_features(state))

    values.append(float(state.get("turn_count", 0)) / 2000.0)
    values.append(float(state.get("bank_houses", 32)) / 32.0)
    values.append(float(state.get("bank_hotels", 12)) / 12.0)

    return np.array(values, dtype=np.float32)


def encode_state_raw(state: Dict[str, Any], num_players: int = 4) -> np.ndarray:
    """
    Estado compacto e menos guiado por heuristicas.

    Este encoder preserva dados observaveis do jogo, mas remove features
    estrategicas derivadas como grupos completos, bloqueios e deltas de troca.
    """

    values: List[float] = []

    current_player = _int_or_default(state.get("current_player", 0), 0)
    values.extend(_one_hot(current_player, 4))

    phase = state.get("phase", "ready_to_roll")
    phase_index = PHASES.index(phase) if phase in PHASES else 0
    values.extend(_one_hot(phase_index, len(PHASES)))

    dice = state.get("dice", (1, 1))
    values.append(float(dice[0]) / 6.0)
    values.append(float(dice[1]) / 6.0)
    values.append(float(state.get("last_roll_total", sum(dice))) / 12.0)
    values.append(float(state.get("consecutive_doubles", 0)) / 3.0)
    values.append(1.0 if state.get("extra_turn_pending", False) else 0.0)

    money = list(state.get("player_money", []))[:4]
    while len(money) < 4:
        money.append(0)
    values.extend([_clamp(float(m) / 3000.0, -1.0, 3.0) for m in money])

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

    owners = list(state.get("property_owners", []))[:NUM_SPACES]
    while len(owners) < NUM_SPACES:
        owners.append(-2)
    for owner in owners:
        if owner == -2:
            values.append(0.0)
        elif owner is None or owner == -1:
            values.append(0.2)
        else:
            values.append((float(owner) + 2.0) / 5.0)

    houses = list(state.get("property_houses", []))[:NUM_SPACES]
    while len(houses) < NUM_SPACES:
        houses.append(0)
    values.extend([float(h) / 5.0 for h in houses])

    mortgaged = list(state.get("property_mortgaged", []))[:NUM_SPACES]
    while len(mortgaged) < NUM_SPACES:
        mortgaged.append(False)
    values.extend([1.0 if m else 0.0 for m in mortgaged])

    context_property = -1
    auction = state.get("auction")
    if isinstance(auction, dict):
        context_property = _int_or_default(auction.get("property_index", -1))
    elif phase == "awaiting_buy" and 0 <= current_player < len(positions):
        context_property = _int_or_default(positions[current_player])
    values.extend(_one_hot(context_property, NUM_SPACES))

    if isinstance(auction, dict):
        values.append(float(auction.get("current_bid", 0)) / 1500.0)
        values.extend(_one_hot(_int_or_default(auction.get("highest_bidder", -1)), 4))
    else:
        values.append(0.0)
        values.extend(_one_hot(-1, 4))

    values.extend(_trade_context_features(state))

    values.append(float(state.get("turn_count", 0)) / 2000.0)
    values.append(float(state.get("action_count", 0)) / 20_000.0)
    values.append(float(state.get("bank_houses", 32)) / 32.0)
    values.append(float(state.get("bank_hotels", 12)) / 12.0)

    return np.array(values, dtype=np.float32)


def encode_action(
    action: Dict[str, Any],
    num_players: int = 4,
    encoder: str | None = DEFAULT_ENCODER,
) -> np.ndarray:
    encoder = normalize_encoder_name(encoder)
    if encoder == ENCODER_RAW:
        return encode_action_raw(action, num_players=num_players)
    return encode_action_rich(action, num_players=num_players)


def encode_action_rich(action: Dict[str, Any], num_players: int = 4) -> np.ndarray:
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
    values.extend(_one_hot(_int_or_default(target_player), 4))

    offer_properties = set(action.get("offer_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in offer_properties else 0.0)

    request_properties = set(action.get("request_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in request_properties else 0.0)

    property_index = _int_or_default(action.get("property_index", -1))
    values.extend(_one_hot(property_index, NUM_SPACES))

    values.append(float(action.get("offer_money", 0)) / 1500.0)
    values.append(float(action.get("request_money", 0)) / 1500.0)
    values.append(float(action.get("amount", 0)) / 1500.0)
    values.extend(_trade_features(action))

    return np.array(values, dtype=np.float32)


def encode_action_raw(action: Dict[str, Any], num_players: int = 4) -> np.ndarray:
    """
    Acao compacta sem features estrategicas de troca.
    """

    values: List[float] = []

    action_type = action.get("type")
    action_index = ACTION_TYPES.index(action_type) if action_type in ACTION_TYPES else -1
    values.extend(_one_hot(action_index, len(ACTION_TYPES)))

    target_player = action.get("target_player", -1)
    values.extend(_one_hot(_int_or_default(target_player), 4))

    offer_properties = set(action.get("offer_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in offer_properties else 0.0)

    request_properties = set(action.get("request_properties", []))
    for i in range(NUM_SPACES):
        values.append(1.0 if i in request_properties else 0.0)

    property_index = _int_or_default(action.get("property_index", -1))
    values.extend(_one_hot(property_index, NUM_SPACES))

    values.append(float(action.get("offer_money", 0)) / 1500.0)
    values.append(float(action.get("request_money", 0)) / 1500.0)
    values.append(float(action.get("amount", 0)) / 1500.0)

    return np.array(values, dtype=np.float32)
