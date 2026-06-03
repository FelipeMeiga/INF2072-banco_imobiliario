from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from game.board import create_board


class Outcome(Enum):
    ONGOING = "ongoing"
    DRAW = "draw"
    WIN1 = "win1"
    WIN2 = "win2"
    WIN3 = "win3"
    WIN4 = "win4"


class Tile(Enum):
    NONE = "none"
    PROPERTY = "property"
    TRAIN = "train"
    UTILITY = "utility"
    CHANCE = "chance"
    CHEST = "chest"
    TAX = "tax"
    JAIL = "jail"


class Card(Enum):
    ADVANCE = "advance"
    RAILROAD2 = "railroad2"
    UTILITY10 = "utility10"
    REWARD = "reward"
    CARD = "card"
    BACK3 = "back3"
    JAIL = "jail"
    REPAIRS = "repairs"
    STREET = "street"
    FINE = "fine"
    CHAIRMAN = "chairman"
    BIRTHDAY = "birthday"


class PlayerState(Enum):
    NORMAL = "normal"
    JAIL = "jail"
    RETIRED = "retired"


PLAYER_COUNT = 4
BANK_INDEX = -1
BOARD_LENGTH = 40
STALEMATE_TURN = 300
GO_BONUS = 200
GO_LANDING_BONUS = 200
JAIL_INDEX = 10
JAIL_PENALTY = 50
MORTGAGE_INTEREST = 1.1

PROPERTY_PENALTIES = [
    [2, 10, 30, 90, 160, 250],
    [4, 20, 60, 180, 320, 450],
    [6, 30, 90, 270, 400, 550],
    [8, 40, 100, 300, 450, 600],
    [10, 50, 150, 450, 625, 750],
    [12, 60, 180, 500, 700, 900],
    [14, 70, 200, 550, 750, 950],
    [16, 80, 220, 600, 800, 1000],
    [18, 90, 250, 700, 875, 1050],
    [20, 100, 300, 750, 925, 1100],
    [22, 110, 330, 800, 975, 1150],
    [22, 120, 360, 850, 1025, 1200],
    [26, 130, 390, 900, 1100, 1275],
    [28, 150, 450, 1000, 1200, 1400],
    [35, 175, 500, 1100, 1300, 1500],
    [50, 200, 600, 1400, 1700, 2000],
]

UTILITY_POSITIONS = [12, 28]
UTILITY_PENALTIES = [4, 10]
TRAIN_POSITIONS = [5, 15, 25, 35]
TRAIN_PENALTIES = [25, 50, 100, 200]

TYPES = [
    Tile.NONE, Tile.PROPERTY, Tile.CHEST, Tile.PROPERTY, Tile.TAX, Tile.TRAIN,
    Tile.PROPERTY, Tile.CHANCE, Tile.PROPERTY, Tile.PROPERTY, Tile.NONE,
    Tile.PROPERTY, Tile.UTILITY, Tile.PROPERTY, Tile.PROPERTY, Tile.TRAIN,
    Tile.PROPERTY, Tile.CHEST, Tile.PROPERTY, Tile.PROPERTY, Tile.NONE,
    Tile.PROPERTY, Tile.CHANCE, Tile.PROPERTY, Tile.PROPERTY, Tile.TRAIN,
    Tile.PROPERTY, Tile.PROPERTY, Tile.UTILITY, Tile.PROPERTY, Tile.JAIL,
    Tile.PROPERTY, Tile.PROPERTY, Tile.CHEST, Tile.PROPERTY, Tile.TRAIN,
    Tile.CHANCE, Tile.PROPERTY, Tile.TAX, Tile.PROPERTY,
]

# Preserves the original repo values, including Dark Blue 1 at 250.
COSTS = [
    0, 60, 0, 60, 200, 200, 100, 0, 100, 120,
    0, 140, 150, 140, 160, 200, 180, 0, 180, 200,
    0, 220, 0, 220, 240, 200, 260, 260, 150, 280,
    0, 300, 300, 0, 320, 200, 0, 250, 100, 400,
]

BUILD = [50, 50, 50, 50, 100, 100, 100, 100, 150, 150, 150, 150, 200, 200, 200, 200]
SETS = [
    [1, 3, -1],
    [6, 8, 9],
    [11, 13, 14],
    [16, 18, 19],
    [21, 23, 24],
    [26, 27, 29],
    [31, 32, 34],
    [37, 39, -1],
]
PROPERTY_INDEX = [-1, 0, -1, 1, -1, -1, 2, -1, 2, 3, -1, 4, -1, 4, 5, -1, 6, -1, 6, 7, -1, 8, -1, 8, 9, -1, 10, 10, -1, 11, -1, 12, 12, -1, 13, -1, -1, 14, -1, 15]
ADAPTER_PROPS = [-1, 0, -1, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, 8, 9, 10, 11, -1, 12, 13, -1, 14, -1, 15, 16, 17, 18, 19, 20, 21, -1, 22, 23, -1, 24, 25, -1, 26, -1, 27]
ADAPTER_HOUSES = [-1, 0, -1, 1, -1, -1, 2, -1, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 9, 10, -1, 11, -1, 12, 13, -1, 14, 15, -1, 16, -1, 17, 18, -1, 19, -1, -1, 20, -1, 21]

GITHUB_INPUT_SIZE = 126
GITHUB_PACK_SIZE = 127
GITHUB_OUTPUT_SIZE = 9
PLAYER_COLORS = [
    (65, 120, 230),
    (230, 60, 60),
    (35, 170, 75),
    (155, 75, 220),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _rand_next(rng: random.Random, minimum: int, maximum: Optional[int] = None) -> int:
    if maximum is None:
        maximum = minimum
        minimum = 0
    if maximum <= minimum:
        return minimum
    return rng.randrange(minimum, maximum)


@dataclass
class CardEntry:
    card: Card
    value: int


class GithubNetworkAdapter:
    def __init__(self):
        self.pack = [0.0] * GITHUB_PACK_SIZE
        self.turn = 0
        self.pos = 4
        self.mon = 8
        self.card = 12
        self.jail = 16
        self.own = 20
        self.mort = 48
        self.house = 76
        self.select = 98
        self.select_money = 126

    def inputs(self, include_money_context: bool = False) -> List[float]:
        size = GITHUB_PACK_SIZE if include_money_context else GITHUB_INPUT_SIZE
        return list(self.pack[:size])

    def reset(self):
        self.pack = [0.0] * GITHUB_PACK_SIZE

    def convert_money(self, money: int) -> float:
        return _clamp(float(money) / 4000.0, 0.0, 1.0)

    def convert_money_value(self, value: float) -> float:
        return value * 4000.0

    def convert_house_value(self, value: float) -> float:
        if value <= 0.5:
            value = 0.0
        return value * 15.0

    def convert_position(self, position: int) -> float:
        return _clamp(float(position) / 39.0, 0.0, 1.0)

    def convert_card(self, cards: int) -> float:
        # Preserves the original bug: it clamps the adapter offset, not cards.
        return _clamp(float(self.card), 0.0, 1.0)

    def convert_house(self, houses: int) -> float:
        return _clamp(float(houses) / 5.0, 0.0, 1.0)

    def set_turn(self, index: int):
        for i in range(4):
            self.pack[i] = 0.0
        self.pack[index] = 1.0

    def set_selection(self, index: int):
        self.clear_selection_state()
        prop = ADAPTER_PROPS[index]
        if prop >= 0:
            self.pack[self.select + prop] = 1.0

    def set_selection_state(self, index: int, state: int):
        prop = ADAPTER_PROPS[index]
        if prop >= 0:
            self.pack[self.select + prop] = float(state)

    def set_money_context(self, state: int):
        self.pack[self.select_money] = float(state)

    def clear_selection_state(self):
        for i in range(self.select, self.select + 29):
            self.pack[i] = 0.0

    def set_position(self, index: int, position: int):
        self.pack[self.pos + index] = self.convert_position(position)

    def set_money(self, index: int, money: int):
        self.pack[self.mon + index] = self.convert_money(money)

    def set_card(self, index: int, cards: int):
        self.pack[self.card + index] = self.convert_card(cards)

    def set_jail(self, index: int, state: int):
        self.pack[self.jail + index] = float(state)

    def set_owner(self, property_index: int, state: int):
        prop = ADAPTER_PROPS[property_index]
        if prop >= 0:
            self.pack[self.own + prop] = (float(state) + 1.0) / 4.0

    def set_mortgage(self, property_index: int, state: int):
        prop = ADAPTER_PROPS[property_index]
        if prop >= 0:
            self.pack[self.mort + prop] = float(state)

    def set_house(self, property_index: int, houses: int):
        house = ADAPTER_HOUSES[property_index]
        if house >= 0:
            self.pack[self.house + house] = self.convert_house(houses)


class GithubBrain:
    def __init__(self, network: Any, include_money_context: bool = False):
        self.network = network
        self.include_money_context = include_money_context
        self.score = 0.0

    def propagate(self, adapter: GithubNetworkAdapter) -> List[float]:
        outputs = self.network.activate(adapter.inputs(self.include_money_context))
        values = [float(value) for value in outputs]
        while len(values) < GITHUB_OUTPUT_SIZE:
            values.append(0.0)
        return values[:GITHUB_OUTPUT_SIZE]


@dataclass
class GithubPlayer:
    brain: GithubBrain
    adapter: GithubNetworkAdapter
    name: str = ""
    color: tuple[int, int, int] = (0, 0, 0)
    state: PlayerState = PlayerState.NORMAL
    position: int = 0
    funds: int = 1500
    jail: int = 0
    doubles: int = 0
    card: int = 0
    items: List[int] = field(default_factory=list)

    @property
    def money(self) -> int:
        return self.funds

    @money.setter
    def money(self, value: int):
        self.funds = value

    @property
    def bankrupt(self) -> bool:
        return self.state == PlayerState.RETIRED

    def decide_buy(self, index: int) -> str:
        return "buy" if self.brain.propagate(self.adapter)[0] > 0.5 else "auction"

    def decide_jail(self) -> str:
        value = self.brain.propagate(self.adapter)[1]
        if value < 0.333:
            return "card"
        if value < 0.666:
            return "roll"
        return "pay"

    def decide_mortgage(self, index: int) -> bool:
        return self.brain.propagate(self.adapter)[2] > 0.5

    def decide_advance(self, index: int) -> bool:
        return self.brain.propagate(self.adapter)[3] > 0.5

    def decide_auction_bid(self, index: int) -> int:
        return int(self.adapter.convert_money_value(self.brain.propagate(self.adapter)[4]))

    def decide_build_house(self, set_index: int) -> int:
        return int(self.adapter.convert_house_value(self.brain.propagate(self.adapter)[5]))

    def decide_sell_house(self, set_index: int) -> int:
        return int(self.adapter.convert_house_value(self.brain.propagate(self.adapter)[6]))

    def decide_offer_trade(self) -> bool:
        return self.brain.propagate(self.adapter)[7] > 0.5

    def decide_accept_trade(self) -> bool:
        return self.brain.propagate(self.adapter)[8] > 0.5


class GithubStyleMonopolyEnv:
    def __init__(
        self,
        networks: List[Any],
        seed: Optional[int] = None,
        include_money_context: bool = False,
        enable_undo: bool = False,
        visual_mode: bool = False,
    ):
        if len(networks) != PLAYER_COUNT:
            raise ValueError("GithubStyleMonopolyEnv exige exatamente 4 redes.")
        self.seed = seed
        self.enable_undo = enable_undo
        self.visual_mode = visual_mode or enable_undo
        self._undo_stack: List[Dict[str, Any]] = []
        self.rng = random.Random(seed)
        self.adapter = GithubNetworkAdapter()
        self.players = [
            GithubPlayer(GithubBrain(network, include_money_context), self.adapter)
            for network in networks
        ]
        self.players = self._shuffle(self.players)
        for i, player in enumerate(self.players):
            player.name = f"Jogador {i + 1}"
            player.color = PLAYER_COLORS[i]
        self.mortgaged = [False] * BOARD_LENGTH
        self.owners = [BANK_INDEX] * BOARD_LENGTH
        self.houses = [0] * BOARD_LENGTH
        self.original = [BANK_INDEX] * BOARD_LENGTH
        self.board = create_board()
        self.turn = 0
        self.count = 0
        self.remaining = PLAYER_COUNT
        self.last_roll = 0
        self.dice = (0, 0)
        self.outcome = Outcome.ONGOING
        self.winner: Optional[int] = None
        self.action_count = 0
        self.event_counter = 0
        self.last_message = "Partida GitHub-style iniciada."
        self.event_history: List[Dict[str, Any]] = []
        self.pending_trade = None
        self.last_trade_result = None
        self.chance = self._make_chance()
        self.chest = self._make_chest()
        for i, player in enumerate(self.players):
            self.adapter.set_position(i, player.position)
            self.adapter.set_money(i, player.funds)
        self._sync_board()

    @property
    def current_player(self) -> GithubPlayer:
        return self.players[self.turn]

    @property
    def current_player_index(self) -> int:
        return self.turn

    @property
    def phase(self) -> str:
        return "game_over" if self.done else "roll"

    @property
    def turn_count(self) -> int:
        return self.count

    @property
    def done(self) -> bool:
        return self.outcome != Outcome.ONGOING

    def get_owned_properties(self, player_index: int) -> List[int]:
        return list(self.players[player_index].items)

    def get_rent_for_space(self, space: Any) -> int:
        try:
            index = self.board.index(space)
        except ValueError:
            return 0

        tile = TYPES[index]
        owner = self.owners[index]
        if tile == Tile.PROPERTY and PROPERTY_INDEX[index] >= 0:
            return PROPERTY_PENALTIES[PROPERTY_INDEX[index]][self.houses[index]]
        if tile == Tile.TRAIN and owner != BANK_INDEX:
            trains = self.count_trains(owner)
            if 1 <= trains <= 4:
                return TRAIN_PENALTIES[trains - 1]
        if tile == Tile.UTILITY and owner != BANK_INDEX:
            utilities = self.count_utilities(owner)
            if 1 <= utilities <= 2:
                return UTILITY_PENALTIES[utilities - 1] * max(1, self.last_roll)
        return 0

    def describe_pending_trade(self) -> List[str]:
        return []

    def undo_last_action(self) -> bool:
        if not self._undo_stack:
            return False
        self._restore_snapshot(self._undo_stack.pop())
        self._sync_board()
        return True

    def _record_event(self, message: str, player_index: Optional[int] = None):
        if not self.visual_mode:
            return
        self.event_counter += 1
        if player_index is None:
            player_index = self.turn
        self.last_message = message
        self.event_history.append(
            {
                "number": self.event_counter,
                "turn": self.count,
                "player": self.players[player_index].name,
                "message": message,
            }
        )
        self.event_history = self.event_history[-80:]

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "mortgaged": list(self.mortgaged),
            "owners": list(self.owners),
            "houses": list(self.houses),
            "original": list(self.original),
            "turn": self.turn,
            "count": self.count,
            "remaining": self.remaining,
            "last_roll": self.last_roll,
            "dice": self.dice,
            "outcome": self.outcome,
            "winner": self.winner,
            "action_count": self.action_count,
            "event_counter": self.event_counter,
            "last_message": self.last_message,
            "event_history": [dict(event) for event in self.event_history],
            "last_trade_result": dict(self.last_trade_result) if self.last_trade_result else None,
            "chance": list(self.chance),
            "chest": list(self.chest),
            "players": [
                {
                    "state": player.state,
                    "position": player.position,
                    "funds": player.funds,
                    "jail": player.jail,
                    "doubles": player.doubles,
                    "card": player.card,
                    "items": list(player.items),
                }
                for player in self.players
            ],
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]):
        self.rng.setstate(snapshot["rng_state"])
        self.mortgaged = list(snapshot["mortgaged"])
        self.owners = list(snapshot["owners"])
        self.houses = list(snapshot["houses"])
        self.original = list(snapshot["original"])
        self.turn = snapshot["turn"]
        self.count = snapshot["count"]
        self.remaining = snapshot["remaining"]
        self.last_roll = snapshot["last_roll"]
        self.dice = tuple(snapshot["dice"])
        self.outcome = snapshot["outcome"]
        self.winner = snapshot["winner"]
        self.action_count = snapshot["action_count"]
        self.event_counter = snapshot["event_counter"]
        self.last_message = snapshot["last_message"]
        self.event_history = [dict(event) for event in snapshot["event_history"]]
        self.last_trade_result = (
            dict(snapshot["last_trade_result"])
            if snapshot["last_trade_result"]
            else None
        )
        self.pending_trade = None
        self.chance = list(snapshot["chance"])
        self.chest = list(snapshot["chest"])

        for player, player_state in zip(self.players, snapshot["players"]):
            player.state = player_state["state"]
            player.position = player_state["position"]
            player.funds = player_state["funds"]
            player.jail = player_state["jail"]
            player.doubles = player_state["doubles"]
            player.card = player_state["card"]
            player.items = list(player_state["items"])

    def _sync_board(self):
        if not self.visual_mode:
            return
        for index, space in enumerate(self.board):
            tile = TYPES[index]
            if tile == Tile.PROPERTY:
                space.type = "property"
            elif tile == Tile.TRAIN:
                space.type = "railroad"
            elif tile == Tile.UTILITY:
                space.type = "utility"
            elif tile == Tile.CHANCE:
                space.type = "chance"
            elif tile == Tile.CHEST:
                space.type = "community_chest"
            elif tile == Tile.TAX:
                space.type = "tax"

            if COSTS[index] > 0:
                space.price = COSTS[index]
                space.mortgage_value = COSTS[index] // 2

            owner = self.owners[index]
            space.owner = None if owner == BANK_INDEX else owner
            space.houses = self.houses[index]
            space.mortgaged = self.mortgaged[index]

    def _shuffle(self, values: List[Any]) -> List[Any]:
        values = list(values)
        self.rng.shuffle(values)
        return values

    def _make_chance(self) -> List[CardEntry]:
        cards = [
            CardEntry(Card.ADVANCE, 39),
            CardEntry(Card.ADVANCE, 0),
            CardEntry(Card.ADVANCE, 24),
            CardEntry(Card.ADVANCE, 11),
            CardEntry(Card.RAILROAD2, 0),
            CardEntry(Card.RAILROAD2, 0),
            CardEntry(Card.UTILITY10, 0),
            CardEntry(Card.REWARD, 50),
            CardEntry(Card.CARD, 0),
            CardEntry(Card.BACK3, 0),
            CardEntry(Card.JAIL, 0),
            CardEntry(Card.REPAIRS, 0),
            CardEntry(Card.FINE, 15),
            CardEntry(Card.ADVANCE, 5),
            CardEntry(Card.CHAIRMAN, 0),
            CardEntry(Card.REWARD, 150),
        ]
        return self._shuffle(cards)

    def _make_chest(self) -> List[CardEntry]:
        cards = [
            CardEntry(Card.ADVANCE, 0),
            CardEntry(Card.REWARD, 200),
            CardEntry(Card.FINE, 50),
            CardEntry(Card.REWARD, 50),
            CardEntry(Card.CARD, 0),
            CardEntry(Card.JAIL, 0),
            CardEntry(Card.REWARD, 100),
            CardEntry(Card.REWARD, 20),
            CardEntry(Card.BIRTHDAY, 0),
            CardEntry(Card.REWARD, 100),
            CardEntry(Card.FINE, 100),
            CardEntry(Card.FINE, 50),
            CardEntry(Card.FINE, 25),
            CardEntry(Card.STREET, 0),
            CardEntry(Card.REWARD, 10),
            CardEntry(Card.REWARD, 100),
        ]
        return self._shuffle(cards)

    def step(self) -> Outcome:
        if self.done:
            return self.outcome
        if self.enable_undo:
            self._undo_stack.append(self._snapshot())
        self.outcome = self.roll()
        self.action_count += 1
        self.winner = self.winner_index(self.outcome)
        if self.outcome == Outcome.DRAW:
            self._record_event("Partida terminou empatada por limite de turnos.")
        elif self.winner is not None:
            self._record_event(f"{self.players[self.winner].name} venceu a partida.", self.winner)
        self._sync_board()
        return self.outcome

    def play(self) -> Outcome:
        outcome = Outcome.ONGOING
        while outcome == Outcome.ONGOING:
            outcome = self.step()
        return outcome

    def roll(self) -> Outcome:
        self.before_turn()
        player = self.players[self.turn]
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.last_roll = d1 + d2
        self.dice = (d1, d2)
        self._record_event(f"{player.name} rolou {d1} + {d2}.")
        is_double = d1 == d2
        double_in_jail = False

        if player.state == PlayerState.JAIL:
            self.adapter.set_turn(self.turn)
            decision = player.decide_jail()
            if decision == "roll":
                if is_double:
                    player.jail = 0
                    player.state = PlayerState.NORMAL
                    self.adapter.set_jail(self.turn, 0)
                    double_in_jail = True
                else:
                    player.jail += 1
                    if player.jail >= 3:
                        self.payment(self.turn, JAIL_PENALTY)
                        player.jail = 0
                        player.state = PlayerState.NORMAL
                        self.adapter.set_jail(self.turn, 0)
            elif decision == "pay":
                self.payment(self.turn, JAIL_PENALTY)
                player.jail = 0
                player.state = PlayerState.NORMAL
                self.adapter.set_jail(self.turn, 0)
            elif decision == "card":
                if player.card > 0:
                    player.card -= 1
                    player.jail = 0
                    player.state = PlayerState.NORMAL
                    self.adapter.set_jail(self.turn, 0)
                    self.adapter.set_card(self.turn, 1 if player.card > 0 else 0)
                elif is_double:
                    player.jail = 0
                    player.state = PlayerState.NORMAL
                    self.adapter.set_jail(self.turn, 0)
                else:
                    player.jail += 1
                    if player.jail >= 3:
                        self.payment(self.turn, JAIL_PENALTY)
                        player.jail = 0
                        player.state = PlayerState.NORMAL
                        self.adapter.set_jail(self.turn, 0)

        if player.state == PlayerState.NORMAL:
            if (not is_double) or player.doubles <= 1:
                self.movement(d1 + d2, is_double)

        if player.state != PlayerState.RETIRED and is_double and not double_in_jail:
            player.doubles += 1
            if player.doubles >= 3:
                player.position = JAIL_INDEX
                player.doubles = 0
                player.state = PlayerState.JAIL
                self.adapter.set_jail(self.turn, 1)
                self._record_event(f"{player.name} tirou tres duplas e foi para a cadeia.")

        return self.end_turn(not is_double or player.state in (PlayerState.RETIRED, PlayerState.JAIL))

    def end_turn(self, increment: bool = True) -> Outcome:
        if increment:
            self.increment_turn()
            skipped = 0
            while self.players[self.turn].state == PlayerState.RETIRED and skipped <= PLAYER_COUNT * 2:
                self.increment_turn()
                skipped += 1
            if self.remaining <= 1:
                return [Outcome.WIN1, Outcome.WIN2, Outcome.WIN3, Outcome.WIN4][self.turn]

        self.count += 1
        if self.count >= STALEMATE_TURN:
            return Outcome.DRAW
        return Outcome.ONGOING

    def increment_turn(self):
        self.turn = (self.turn + 1) % PLAYER_COUNT

    def before_turn(self):
        player = self.players[self.turn]
        if player.state == PlayerState.RETIRED:
            return

        for index in list(player.items):
            if self.mortgaged[index]:
                advance_price = int(COSTS[index] * MORTGAGE_INTEREST)
                if advance_price > player.funds:
                    continue
                self.adapter.set_turn(self.turn)
                self.adapter.set_selection_state(index, 1)
                decision = player.decide_advance(index)
                self.adapter.set_selection_state(index, 0)
                if decision:
                    self.advance(index)
            else:
                self.adapter.set_turn(self.turn)
                self.adapter.set_selection_state(index, 1)
                decision = player.decide_mortgage(index)
                self.adapter.set_selection_state(index, 0)
                if decision:
                    # The original has Mortgage(index) commented out here.
                    pass

        self._sell_then_build_sets(self.turn, free_phase=True)
        self.trading()

    def _sell_then_build_sets(self, owner: int, free_phase: bool):
        sets = self.find_sets(owner)
        for set_index in sets:
            house_total = self._set_house_total(set_index)
            sell_max = house_total
            self.adapter.set_turn(owner)
            self.adapter.set_selection_state(SETS[set_index][0], 1)
            decision = self.players[owner].decide_sell_house(set_index)
            self.adapter.set_selection_state(SETS[set_index][0], 0)
            decision = min(decision, sell_max)
            if decision > 0:
                self.sell_houses(set_index, decision)
                self.players[owner].funds += int(decision * BUILD[PROPERTY_INDEX[SETS[set_index][0]]] * 0.5)
                self.adapter.set_money(owner, self.players[owner].funds)
                self._record_event(
                    f"{self.players[owner].name} vendeu {decision} construcao(oes).",
                    owner,
                )

        sets = self.find_sets(owner)
        for set_index in sets:
            max_house = 10 if set_index in (0, 7) else 15
            house_total = self._set_house_total(set_index)
            build_max = max_house - house_total
            afford_max = int(self.players[owner].funds // float(BUILD[PROPERTY_INDEX[SETS[set_index][0]]]))
            if afford_max < 0:
                afford_max = 0
            build_max = min(build_max, afford_max)
            self.adapter.set_turn(owner)
            self.adapter.set_selection_state(SETS[set_index][0], 1)
            decision = self.players[owner].decide_build_house(set_index)
            self.adapter.set_selection_state(SETS[set_index][0], 0)
            decision = min(decision, build_max)
            if decision > 0:
                self.build_houses(set_index, decision)
                self.payment(owner, decision * BUILD[PROPERTY_INDEX[SETS[set_index][0]]])
                self._record_event(
                    f"{self.players[owner].name} construiu {decision} casa(s).",
                    owner,
                )

    def _set_house_total(self, set_index: int) -> int:
        total = self.houses[SETS[set_index][0]] + self.houses[SETS[set_index][1]]
        if set_index not in (0, 7):
            total += self.houses[SETS[set_index][2]]
        return total

    def trading(self):
        candidates = []
        candidate_indexes = []
        for i, other in enumerate(self.players):
            if i == self.turn or other.state == PlayerState.RETIRED:
                continue
            candidates.append(other)
            candidate_indexes.append(i)
        if not candidates:
            return

        trade_attempts = 4
        trade_item_max = 5
        trade_money_max = 500
        player = self.players[self.turn]
        for _ in range(trade_attempts):
            give = _rand_next(self.rng, 0, min(len(player.items), trade_item_max))
            selected_player = _rand_next(self.rng, 0, len(candidates))
            other = candidates[selected_player]
            other_index = candidate_indexes[selected_player]
            receive = _rand_next(self.rng, 0, min(len(other.items), trade_item_max))

            if player.funds < 0 or other.funds < 0:
                continue
            money_give = _rand_next(self.rng, 0, min(player.funds, trade_money_max))
            money_receive = _rand_next(self.rng, 0, min(other.funds, trade_money_max))
            money_balance = money_give - money_receive
            if give == 0 or receive == 0:
                continue

            gift = []
            possible = list(player.items)
            for _ in range(give):
                selection = _rand_next(self.rng, 0, len(possible))
                gift.append(possible.pop(selection))

            returning = []
            possible = list(other.items)
            for _ in range(receive):
                selection = _rand_next(self.rng, 0, len(possible))
                returning.append(possible.pop(selection))

            for item in gift:
                self.adapter.set_selection_state(item, 1)
            for item in returning:
                self.adapter.set_selection_state(item, 1)
            self.adapter.set_money_context(money_balance)

            if not player.decide_offer_trade():
                self.last_trade_result = self._make_trade_result(
                    self.turn,
                    other_index,
                    gift,
                    returning,
                    money_give,
                    money_receive,
                    "declined",
                )
                self.adapter.clear_selection_state()
                continue
            if not other.decide_accept_trade():
                self.last_trade_result = self._make_trade_result(
                    self.turn,
                    other_index,
                    gift,
                    returning,
                    money_give,
                    money_receive,
                    "declined",
                )
                self._record_event(
                    f"{other.name} recusou troca de {player.name}.",
                    other_index,
                )
                continue

            for item in gift:
                player.items.remove(item)
                other.items.append(item)
                self.owners[item] = other_index
                self.adapter.set_owner(item, other_index)

            for item in returning:
                other.items.remove(item)
                player.items.append(item)
                self.owners[item] = self.turn
                self.adapter.set_owner(item, self.turn)

            self.adapter.clear_selection_state()
            player.funds -= money_balance
            other.funds += money_balance
            self.adapter.set_money(self.turn, player.funds)
            self.adapter.set_money(other_index, other.funds)
            self.last_trade_result = self._make_trade_result(
                self.turn,
                other_index,
                gift,
                returning,
                money_give,
                money_receive,
                "accepted",
            )
            self._record_event(
                f"{player.name} e {other.name} fecharam uma troca.",
                self.turn,
            )

    def _make_trade_result(
        self,
        from_player: int,
        to_player: int,
        offer_properties: List[int],
        request_properties: List[int],
        offer_money: int,
        request_money: int,
        status: str,
    ) -> Dict[str, Any]:
        return {
            "from": from_player,
            "to": to_player,
            "offer_properties": list(offer_properties),
            "request_properties": list(request_properties),
            "offer_money": offer_money,
            "request_money": request_money,
            "status": status,
        }

    def auction(self, index: int):
        participation = [player.state != PlayerState.RETIRED for player in self.players]
        bids = [0] * PLAYER_COUNT
        for i, player in enumerate(self.players):
            self.adapter.set_turn(i)
            self.adapter.set_selection_state(index, 1)
            bids[i] = player.decide_auction_bid(index)
            self.adapter.set_selection_state(index, 0)
            if bids[i] > player.funds:
                participation[i] = False

        max_bid = 0
        for i in range(PLAYER_COUNT):
            if participation[i] and bids[i] > max_bid:
                max_bid = bids[i]

        candidates = [i for i in range(PLAYER_COUNT) if participation[i] and bids[i] == max_bid]
        backup = [i for i, player in enumerate(self.players) if player.state != PlayerState.RETIRED]
        if candidates:
            winner = candidates[_rand_next(self.rng, 0, len(candidates))]
            self.payment(winner, max_bid)
        else:
            winner = backup[_rand_next(self.rng, 0, len(backup))]

        self.owners[index] = winner
        self.players[winner].items.append(index)
        if self.original[index] == BANK_INDEX:
            self.original[index] = winner
        self.adapter.set_owner(index, winner)
        self._record_event(
            f"{self.players[winner].name} venceu leilao de {self.board[index].name} por ${max_bid}.",
            winner,
        )

    def movement(self, roll: int, is_double: bool):
        player = self.players[self.turn]
        player.position += roll
        if player.position >= BOARD_LENGTH:
            player.position -= BOARD_LENGTH
            if player.position == 0:
                player.funds += GO_BONUS
            else:
                player.funds += GO_LANDING_BONUS
        self.adapter.set_money(self.turn, player.funds)
        self.adapter.set_position(self.turn, player.position)
        self.activate_tile()

    def activate_tile(self):
        index = self.players[self.turn].position
        tile = TYPES[index]
        if tile == Tile.PROPERTY:
            self._activate_ownable(index, rent_kind="property")
        elif tile == Tile.TRAIN:
            self._activate_ownable(index, rent_kind="train")
        elif tile == Tile.UTILITY:
            self._activate_ownable(index, rent_kind="utility")
        elif tile == Tile.TAX:
            self.payment(self.turn, COSTS[index])
            self._record_event(f"{self.players[self.turn].name} pagou imposto de ${COSTS[index]}.")
        elif tile == Tile.CHANCE:
            self._record_event(f"{self.players[self.turn].name} caiu em Chance.")
            self.draw_chance()
        elif tile == Tile.CHEST:
            self._record_event(f"{self.players[self.turn].name} caiu em Community Chest.")
            self.draw_chest()
        elif tile == Tile.JAIL:
            player = self.players[self.turn]
            player.position = JAIL_INDEX
            player.doubles = 0
            player.state = PlayerState.JAIL
            self.adapter.set_jail(self.turn, 1)
            self._record_event(f"{player.name} foi para a cadeia.")

    def _activate_ownable(self, index: int, rent_kind: str):
        owner = self.owner(index)
        player = self.players[self.turn]
        if owner == BANK_INDEX:
            self.adapter.set_turn(self.turn)
            self.adapter.set_selection(index)
            decision = player.decide_buy(index)
            if decision == "buy":
                if player.funds < COSTS[index]:
                    self.auction(index)
                else:
                    self.payment(self.turn, COSTS[index])
                    self.owners[index] = self.turn
                    if self.original[index] == BANK_INDEX:
                        self.original[index] = self.turn
                    player.items.append(index)
                    # The original property branch accidentally writes owner=-1.
                    if rent_kind == "property":
                        self.adapter.set_owner(index, owner)
                    else:
                        self.adapter.set_owner(index, self.turn)
                    self._record_event(
                        f"{player.name} comprou {self.board[index].name} por ${COSTS[index]}.",
                        self.turn,
                    )
            elif decision == "auction":
                self._record_event(f"{player.name} mandou {self.board[index].name} para leilao.")
                self.auction(index)
            return

        if owner == self.turn or self.mortgaged[index]:
            return

        if rent_kind == "property":
            fine = PROPERTY_PENALTIES[PROPERTY_INDEX[index]][self.houses[index]]
            self.payment_to_player(self.turn, owner, fine)
            self._record_event(
                f"{player.name} pagou ${fine} de aluguel para {self.players[owner].name}.",
                self.turn,
            )
        elif rent_kind == "train":
            trains = self.count_trains(owner)
            if 1 <= trains <= 4:
                fine = TRAIN_PENALTIES[trains - 1]
                self.payment_to_player(self.turn, owner, fine)
                self._record_event(
                    f"{player.name} pagou ${fine} de ferrovia para {self.players[owner].name}.",
                    self.turn,
                )
        elif rent_kind == "utility":
            utilities = self.count_utilities(owner)
            if 1 <= utilities <= 2:
                fine = UTILITY_PENALTIES[utilities - 1] * self.last_roll
                self.payment_to_player(self.turn, owner, fine)
                self._record_event(
                    f"{player.name} pagou ${fine} de companhia para {self.players[owner].name}.",
                    self.turn,
                )

    def payment(self, owner: int, fine: int):
        self.players[owner].funds -= fine
        self.adapter.set_money(owner, self.players[owner].funds)
        self._handle_debt_to_bank(owner)

    def payment_to_player(self, owner: int, recipient: int, fine: int):
        self.players[owner].funds -= fine
        self.adapter.set_money(owner, self.players[owner].funds)
        self.players[recipient].funds += fine
        self.adapter.set_money(recipient, self.players[recipient].funds)
        self._handle_debt_to_player(owner, recipient)

    def _handle_debt_to_bank(self, owner: int):
        if self.players[owner].funds < 0:
            self._prompt_sell_sets(owner)
        if self.players[owner].funds < 0:
            self._prompt_mortgages(owner)
        if self.players[owner].funds < 0:
            for item in list(self.players[owner].items):
                self.owners[item] = BANK_INDEX
                self.adapter.set_owner(item, BANK_INDEX)
                self.houses[item] = 0
                self.adapter.set_house(item, 0)
            self.players[owner].items.clear()
            self.players[owner].state = PlayerState.RETIRED
            self.remaining -= 1
            self._record_event(f"{self.players[owner].name} faliu para o banco.", owner)

    def _handle_debt_to_player(self, owner: int, recipient: int):
        if self.players[owner].funds < 0:
            self._prompt_sell_sets(owner)
        if self.players[owner].funds < 0:
            self._prompt_mortgages(owner)
        if self.players[owner].funds < 0:
            self.players[recipient].funds += self.players[owner].funds
            self.adapter.set_money(recipient, self.players[recipient].funds)
            house_money = 0
            for item in list(self.players[owner].items):
                self.players[recipient].items.append(item)
                self.owners[item] = recipient
                self.adapter.set_owner(item, recipient)
                if self.houses[item] > 0:
                    house_money += (self.houses[item] * BUILD[PROPERTY_INDEX[item]]) // 2
                    self.houses[item] = 0
                    self.adapter.set_house(item, 0)
            self.players[recipient].funds += house_money
            self.adapter.set_money(recipient, self.players[recipient].funds)
            self.players[owner].items.clear()
            self.players[owner].state = PlayerState.RETIRED
            self.remaining -= 1
            self._record_event(
                f"{self.players[owner].name} faliu para {self.players[recipient].name}.",
                owner,
            )

    def _prompt_sell_sets(self, owner: int):
        for set_index in self.find_sets(owner):
            sell_max = self._set_house_total(set_index)
            self.adapter.set_turn(owner)
            self.adapter.set_selection_state(SETS[set_index][0], 1)
            decision = self.players[owner].decide_sell_house(set_index)
            self.adapter.set_selection_state(SETS[set_index][0], 0)
            decision = min(decision, sell_max)
            if decision > 0:
                self.sell_houses(set_index, decision)
                self.players[owner].funds += int(decision * BUILD[PROPERTY_INDEX[SETS[set_index][0]]] * 0.5)
                self.adapter.set_money(owner, self.players[owner].funds)

    def _prompt_mortgages(self, owner: int):
        for item in list(self.players[owner].items):
            self.adapter.set_turn(owner)
            self.adapter.set_selection_state(item, 1)
            decision = self.players[owner].decide_mortgage(item)
            self.adapter.set_selection_state(item, 0)
            if decision:
                self.mortgage(item)

    def owner(self, index: int) -> int:
        return self.owners[index]

    def mortgage(self, index: int):
        self.mortgaged[index] = True
        self.adapter.set_mortgage(index, 1)
        owner = self.owners[index]
        self.players[owner].funds += COSTS[index] // 2
        self.adapter.set_money(owner, self.players[owner].funds)
        self._record_event(
            f"{self.players[owner].name} hipotecou {self.board[index].name} por ${COSTS[index] // 2}.",
            owner,
        )

    def advance(self, index: int):
        self.mortgaged[index] = False
        self.adapter.set_mortgage(index, 0)
        self.payment(self.owners[index], int(COSTS[index] * MORTGAGE_INTEREST))

    def count_trains(self, player: int) -> int:
        return sum(1 for item in self.players[player].items if item in TRAIN_POSITIONS)

    def count_utilities(self, player: int) -> int:
        return sum(1 for item in self.players[player].items if item in UTILITY_POSITIONS)

    def draw_chance(self):
        card = self.chance.pop(0)
        self.chance.append(card)
        self._apply_card(card, chance=True)

    def draw_chest(self):
        card = self.chest.pop(0)
        self.chest.append(card)
        self._apply_card(card, chance=False)

    def _apply_card(self, card: CardEntry, chance: bool):
        player = self.players[self.turn]
        if card.card == Card.ADVANCE:
            if player.position > card.value:
                player.funds += GO_BONUS
                self.adapter.set_money(self.turn, player.funds)
            player.position = card.value
            self.adapter.set_position(self.turn, player.position)
            self.activate_tile()
        elif card.card == Card.REWARD:
            player.funds += card.value
            self.adapter.set_money(self.turn, player.funds)
        elif card.card == Card.FINE:
            self.payment(self.turn, card.value)
        elif card.card == Card.BACK3:
            player.position -= 3
            self.adapter.set_position(self.turn, player.position)
            self.activate_tile()
        elif card.card == Card.CARD:
            player.card += 1
            self.adapter.set_card(self.turn, player.card)
        elif card.card == Card.JAIL:
            player.position = JAIL_INDEX
            player.doubles = 0
            player.state = PlayerState.JAIL
            self.adapter.set_position(self.turn, player.position)
            self.adapter.set_jail(self.turn, 1)
        elif card.card == Card.RAILROAD2:
            self.advance_to_train2()
        elif card.card == Card.UTILITY10:
            self.advance_to_utility10()
        elif card.card == Card.CHAIRMAN:
            for i, other in enumerate(self.players):
                if i != self.turn and other.state != PlayerState.RETIRED:
                    self.payment_to_player(self.turn, i, 50)
        elif card.card == Card.REPAIRS:
            house_count, hotel_count = self._building_counts(self.turn)
            self.payment(self.turn, house_count * 25 + hotel_count * 100)
        elif card.card == Card.BIRTHDAY:
            for i, other in enumerate(self.players):
                if i != self.turn and other.state != PlayerState.RETIRED:
                    self.payment_to_player(i, self.turn, 10)
        elif card.card == Card.STREET:
            house_count, hotel_count = self._building_counts(self.turn)
            self.payment(self.turn, house_count * 40 + hotel_count * 115)

    def _building_counts(self, player_index: int) -> tuple[int, int]:
        houses = 0
        hotels = 0
        for item in self.players[player_index].items:
            if self.houses[item] <= 4:
                houses += self.houses[item]
            else:
                hotels += 1
        return houses, hotels

    def advance_to_train2(self):
        index = self.players[self.turn].position
        targets = [pos for pos in TRAIN_POSITIONS if index < pos]
        if targets:
            self.players[self.turn].position = targets[0]
        else:
            self.players[self.turn].position = TRAIN_POSITIONS[0]
            self.players[self.turn].funds += GO_BONUS
            self.adapter.set_money(self.turn, self.players[self.turn].funds)
        self.adapter.set_position(self.turn, self.players[self.turn].position)
        self._activate_nearest_purchase(multiplier_kind="train2")

    def advance_to_utility10(self):
        index = self.players[self.turn].position
        targets = [pos for pos in UTILITY_POSITIONS if index < pos]
        if targets:
            self.players[self.turn].position = targets[0]
        else:
            self.players[self.turn].position = UTILITY_POSITIONS[0]
            self.players[self.turn].funds += GO_BONUS
            self.adapter.set_money(self.turn, self.players[self.turn].funds)
        self.adapter.set_position(self.turn, self.players[self.turn].position)
        self._activate_nearest_purchase(multiplier_kind="utility10")

    def _activate_nearest_purchase(self, multiplier_kind: str):
        index = self.players[self.turn].position
        owner = self.owner(index)
        player = self.players[self.turn]
        if owner == BANK_INDEX:
            self.adapter.set_turn(self.turn)
            decision = player.decide_buy(index)
            if decision == "buy":
                if player.funds < COSTS[index]:
                    self.auction(index)
                else:
                    self.payment(self.turn, COSTS[index])
                    self.owners[index] = self.turn
                    if self.original[index] == BANK_INDEX:
                        self.original[index] = self.turn
                    player.items.append(index)
                    self.adapter.set_owner(index, self.turn)
            elif decision == "auction":
                self.auction(index)
        elif owner != self.turn and not self.mortgaged[index]:
            if multiplier_kind == "train2":
                trains = self.count_trains(owner)
                if 1 <= trains <= 4:
                    self.payment_to_player(self.turn, owner, TRAIN_PENALTIES[trains - 1] * 2)
            elif multiplier_kind == "utility10":
                self.payment_to_player(self.turn, owner, 10 * self.last_roll)

    def find_sets(self, owner: int) -> List[int]:
        items = self.players[owner].items
        sets = []
        for i, group in enumerate(SETS):
            if i in (0, 7):
                if group[0] in items and group[1] in items:
                    sets.append(i)
            elif group[0] in items and group[1] in items and group[2] in items:
                sets.append(i)
        return sets

    def build_houses(self, set_index: int, amount: int):
        last = 1 if set_index in (0, 7) else 2
        for _ in range(amount):
            bj = last
            for j in range(last - 1, -1, -1):
                if self.houses[SETS[set_index][bj]] > self.houses[SETS[set_index][j]]:
                    bj = j
            prop = SETS[set_index][bj]
            self.houses[prop] += 1
            self.adapter.set_house(prop, self.houses[prop])

    def sell_houses(self, set_index: int, amount: int):
        last = 1 if set_index in (0, 7) else 2
        for _ in range(amount):
            bj = 0
            for j in range(0, last + 1):
                if self.houses[SETS[set_index][bj]] < self.houses[SETS[set_index][j]]:
                    bj = j
            prop = SETS[set_index][bj]
            self.houses[prop] -= 1
            self.adapter.set_house(prop, self.houses[prop])

    def winner_index(self, outcome: Outcome) -> Optional[int]:
        mapping = {
            Outcome.WIN1: 0,
            Outcome.WIN2: 1,
            Outcome.WIN3: 2,
            Outcome.WIN4: 3,
        }
        return mapping.get(outcome)

    def net_worth(self, index: int) -> int:
        player = self.players[index]
        total = player.funds
        for item in player.items:
            total += COSTS[item]
            total += self.houses[item] * (BUILD[PROPERTY_INDEX[item]] if PROPERTY_INDEX[item] >= 0 else 0)
        return total
