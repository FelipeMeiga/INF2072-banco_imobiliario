import copy
import math
import random
from typing import Any, Dict, List, Optional

from game.actions import (
    ACCEPT_TRADE,
    ADD_TRADE_OFFER_PROPERTY,
    ADD_TRADE_REQUEST_PROPERTY,
    AUCTION_BID,
    AUCTION_PASS,
    BUILD_HOUSE,
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
    SELL_HOUSE,
    SELECT_TRADE_TARGET,
    SET_TRADE_OFFER_MONEY,
    SET_TRADE_REQUEST_MONEY,
    START_TRADE,
    SUBMIT_TRADE,
    UNMORTGAGE_PROPERTY,
    USE_JAIL_CARD,
)
from game.board import (
    BANK_HOTELS,
    BANK_HOUSES,
    GO_INDEX,
    GO_TO_JAIL_INDEX,
    GROUPS,
    JAIL_INDEX,
    MAX_HOUSES,
    NUM_SPACES,
    RAILROADS,
    UTILITIES,
    create_board,
)
from game.models import Player, Space

START_MONEY = 1500
PASS_START_BONUS = 200
JAIL_FINE = 50
MAX_TURNS = 2000
DEFENSIVE_CASH_THRESHOLD = 0
UNMORTGAGE_CASH_RESERVE = 300
TRADE_MAX_PROPERTIES_PER_SIDE = 3
TRADE_MONEY_CANDIDATES = (0, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500)
CRITICAL_BUY_CASH_RESERVE = 75
AUCTION_BLOCK_VALUE_MULTIPLIER = 2.25
AUCTION_COMPLETE_VALUE_MULTIPLIER = 1.90

PLAYER_COLORS = [
    (70, 120, 220),
    (210, 60, 60),
    (50, 170, 80),
    (150, 80, 200),
]


class BancoImobiliarioEnv:
    """
    Rules-compatible Monopoly-style engine with generic board labels.

    The branded board names/artwork are intentionally not copied, but the major
    classic mechanics are represented: auctions, doubles, jail, mortgages, color
    sets, even building, limited houses/hotels, cards, trades, utilities and
    railroads.
    """

    def __init__(
        self,
        num_players: int = 4,
        seed: Optional[int] = None,
        enable_undo: bool = False,
        max_undo_history: int = 200,
    ):
        self.num_players = num_players
        self.seed = seed
        self.enable_undo = enable_undo
        self.max_undo_history = max_undo_history
        self.undo_history: List[Dict[str, Any]] = []
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.board: List[Space] = create_board(self.seed)
        self.players: List[Player] = [
            Player(
                name=f"Jogador {i + 1}",
                color=PLAYER_COLORS[i % len(PLAYER_COLORS)],
                money=START_MONEY,
            )
            for i in range(self.num_players)
        ]

        self.current_player_index = self.random.randrange(self.num_players)
        self.dice = (1, 1)
        self.last_roll_total = 2
        self.consecutive_doubles = 0
        self.extra_turn_pending = False
        self.phase = "ready_to_roll"
        self.trade_draft: Optional[Dict[str, Any]] = None
        self.pending_trade: Optional[Dict[str, Any]] = None
        self.last_trade_result: Optional[Dict[str, Any]] = None
        self.auction: Optional[Dict[str, Any]] = None
        self.trade_proposed_this_turn = False
        self.properties_sold_buildings_this_turn: set[int] = set()
        self.properties_built_this_turn: set[int] = set()
        self.bank_houses = BANK_HOUSES
        self.bank_hotels = BANK_HOTELS
        self.chance_deck = self._make_chance_deck()
        self.community_deck = self._make_community_deck()
        self.done = False
        self.winner: Optional[int] = None
        self.turn_count = 0
        self.action_count = 0
        self.last_message = f"Jogo iniciado. {self.current_player.name} comeca."
        self.last_action: Optional[Dict[str, Any]] = None
        self.event_history: List[Dict[str, Any]] = []
        self.undo_history = []
        return self.get_state()

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_player": self.current_player_index,
            "phase": self.phase,
            "dice": self.dice,
            "last_roll_total": self.last_roll_total,
            "consecutive_doubles": self.consecutive_doubles,
            "extra_turn_pending": self.extra_turn_pending,
            "player_money": [p.money for p in self.players],
            "player_positions": [p.position for p in self.players],
            "player_bankrupt": [p.bankrupt for p in self.players],
            "player_in_jail": [p.in_jail for p in self.players],
            "player_jail_turns": [p.jail_turns for p in self.players],
            "player_jail_free_cards": [p.jail_free_cards for p in self.players],
            "property_owners": [
                space.owner if space.is_ownable() else -2
                for space in self.board
            ],
            "property_houses": [
                space.houses if space.type == "property" else 0
                for space in self.board
            ],
            "property_mortgaged": [
                space.mortgaged if space.is_ownable() else False
                for space in self.board
            ],
            "property_groups": [
                space.group if space.type == "property" else None
                for space in self.board
            ],
            "player_net_worth": [self._net_worth(i) for i in range(self.num_players)],
            "player_completed_groups": [self._completed_group_count(i) for i in range(self.num_players)],
            "player_near_groups": [self._near_group_count(i) for i in range(self.num_players)],
            "player_blocking_properties": [self._blocking_property_count(i) for i in range(self.num_players)],
            "player_highest_rent_potential": [self._highest_rent_potential(i) for i in range(self.num_players)],
            "player_group_progress": [self._best_group_progress(i) for i in range(self.num_players)],
            "trade_draft": copy.deepcopy(self.trade_draft),
            "pending_trade": copy.deepcopy(self.pending_trade),
            "last_trade_result": copy.deepcopy(self.last_trade_result),
            "auction": copy.deepcopy(self.auction),
            "bank_houses": self.bank_houses,
            "bank_hotels": self.bank_hotels,
            "done": self.done,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "action_count": self.action_count,
            "undo_count": len(self.undo_history),
            "last_message": self.last_message,
            "event_history": self.event_history.copy(),
        }

    def get_valid_actions(self, player_index: Optional[int] = None) -> List[Dict[str, Any]]:
        if player_index is None:
            player_index = self.current_player_index

        if self.done or self.players[player_index].bankrupt:
            return []

        if self.phase == "auction":
            if not self.auction or self.auction["current_bidder"] != player_index:
                return []
            return self._get_auction_actions(player_index)

        if self.phase == "pending_trade_response":
            if self.pending_trade and self.pending_trade["to"] == player_index:
                return [
                    self._trade_response_action(ACCEPT_TRADE),
                    self._trade_response_action(DECLINE_TRADE),
                ]
            return []

        if self.phase == "building_trade":
            if self.trade_draft and self.trade_draft["from"] == player_index:
                return self._get_trade_draft_actions(player_index)
            return []

        if player_index != self.current_player_index:
            return []

        if self.phase == "awaiting_buy":
            if self._should_force_buy_property(player_index):
                return [{"type": BUY_PROPERTY}]
            return [{"type": BUY_PROPERTY}, {"type": PASS_BUY}]

        if self.phase == "ready_to_roll":
            actions = []
            player = self.players[player_index]

            if player.in_jail:
                actions.append({"type": ROLL_DICE})
                if player.money >= JAIL_FINE:
                    actions.append({"type": PAY_JAIL_FINE})
                if player.jail_free_cards > 0:
                    actions.append({"type": USE_JAIL_CARD})
            else:
                actions.append({"type": ROLL_DICE})

            actions.extend(self._get_financial_actions(player_index))

            if not self.trade_proposed_this_turn and self._can_start_trade(player_index):
                actions.append({"type": START_TRADE})

            return actions

        return []

    def recover_no_action_state(self, player_index: Optional[int] = None) -> bool:
        if self.done:
            return False

        active_players = [i for i, player in enumerate(self.players) if not player.bankrupt]
        if len(active_players) <= 1:
            self._check_game_end()
            return True

        if self.phase == "pending_trade_response":
            if self.pending_trade is None or self.players[self.pending_trade["to"]].bankrupt:
                self._clear_trade()
                return True

        if self.phase == "building_trade":
            if self.trade_draft is None or self.players[self.trade_draft["from"]].bankrupt:
                self._clear_trade_draft()
                return True

        if self.phase == "auction":
            return self._recover_auction_state()

        if player_index is None:
            player_index = self.current_player_index

        if self.players[player_index].bankrupt and player_index == self.current_player_index:
            self._next_turn()
            return True

        return False

    def finish_interrupted_game(self):
        if not self.done:
            self._finish_by_net_worth()

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.get_state(), 0.0, self.done, {"message": self.last_message}

        self._push_undo_snapshot()
        self.last_action = action
        acting_player_index = self._get_acting_player_for_action(action)
        before_net_worth = self._reward_net_worth(acting_player_index)
        before_completed_groups = self._completed_group_count(acting_player_index)
        before_total_houses = self._total_houses(acting_player_index)
        purchase_before_action = self._current_purchase_property_index()
        auction_before_action = copy.deepcopy(self.auction)
        action_type = action.get("type")

        if action_type == ROLL_DICE:
            self._roll_dice()
        elif action_type == BUY_PROPERTY:
            self._buy_property()
        elif action_type == PASS_BUY:
            self._pass_buy()
        elif action_type == AUCTION_BID:
            self._auction_bid(action)
        elif action_type == AUCTION_PASS:
            self._auction_pass()
        elif action_type == PAY_JAIL_FINE:
            self._pay_jail_fine()
        elif action_type == USE_JAIL_CARD:
            self._use_jail_card()
        elif action_type == START_TRADE:
            self._start_trade()
        elif action_type == SELECT_TRADE_TARGET:
            self._select_trade_target(action)
        elif action_type == ADD_TRADE_OFFER_PROPERTY:
            self._add_trade_offer_property(action)
        elif action_type == FINISH_TRADE_OFFER:
            self._finish_trade_offer()
        elif action_type == SET_TRADE_OFFER_MONEY:
            self._set_trade_offer_money(action)
        elif action_type == ADD_TRADE_REQUEST_PROPERTY:
            self._add_trade_request_property(action)
        elif action_type == FINISH_TRADE_REQUEST:
            self._finish_trade_request()
        elif action_type == SET_TRADE_REQUEST_MONEY:
            self._set_trade_request_money(action)
        elif action_type == SUBMIT_TRADE:
            self._submit_trade()
        elif action_type == CANCEL_TRADE:
            self._cancel_trade()
        elif action_type == PROPOSE_TRADE:
            self._propose_trade(action)
        elif action_type == ACCEPT_TRADE:
            self._accept_trade()
        elif action_type == DECLINE_TRADE:
            self._decline_trade()
        elif action_type == BUILD_HOUSE:
            self._build_house(action)
        elif action_type == SELL_HOUSE:
            self._sell_house(action)
        elif action_type == MORTGAGE_PROPERTY:
            self._mortgage_property(action)
        elif action_type == UNMORTGAGE_PROPERTY:
            self._unmortgage_property(action)
        else:
            self.last_message = f"Acao invalida: {action_type}"

        self._check_game_end()

        after_net_worth = self._reward_net_worth(acting_player_index)
        after_completed_groups = self._completed_group_count(acting_player_index)
        after_total_houses = self._total_houses(acting_player_index)
        reward = (after_net_worth - before_net_worth) / 100.0
        reward += self._strategic_reward(
            action_type,
            before_completed_groups,
            after_completed_groups,
            before_total_houses,
            after_total_houses,
        )
        reward += self._purchase_auction_strategy_reward(
            action_type,
            acting_player_index,
            action,
            purchase_before_action,
            auction_before_action,
        )

        if self.done and self.winner is not None:
            reward += 100.0 if self.winner == acting_player_index else -100.0

        self._register_event(acting_player_index, action, reward)
        return self.get_state(), reward, self.done, {"message": self.last_message}

    # ========================================================
    # Undo / replay support
    # ========================================================

    def undo_last_action(self) -> bool:
        if not self.undo_history:
            self.last_message = "Nao ha jogada anterior para voltar."
            return False

        snapshot = self.undo_history.pop()
        self._restore_snapshot(snapshot)
        self.last_message = "Jogada anterior restaurada."
        return True

    def _push_undo_snapshot(self):
        if not self.enable_undo:
            return

        self.undo_history.append(self._capture_snapshot())
        if len(self.undo_history) > self.max_undo_history:
            self.undo_history = self.undo_history[-self.max_undo_history:]

    def _capture_snapshot(self) -> Dict[str, Any]:
        return {
            "board": copy.deepcopy(self.board),
            "players": copy.deepcopy(self.players),
            "current_player_index": self.current_player_index,
            "dice": self.dice,
            "last_roll_total": self.last_roll_total,
            "consecutive_doubles": self.consecutive_doubles,
            "extra_turn_pending": self.extra_turn_pending,
            "phase": self.phase,
            "trade_draft": copy.deepcopy(self.trade_draft),
            "pending_trade": copy.deepcopy(self.pending_trade),
            "last_trade_result": copy.deepcopy(self.last_trade_result),
            "auction": copy.deepcopy(self.auction),
            "trade_proposed_this_turn": self.trade_proposed_this_turn,
            "properties_sold_buildings_this_turn": copy.deepcopy(self.properties_sold_buildings_this_turn),
            "properties_built_this_turn": copy.deepcopy(self.properties_built_this_turn),
            "bank_houses": self.bank_houses,
            "bank_hotels": self.bank_hotels,
            "chance_deck": copy.deepcopy(self.chance_deck),
            "community_deck": copy.deepcopy(self.community_deck),
            "done": self.done,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "action_count": self.action_count,
            "last_message": self.last_message,
            "last_action": copy.deepcopy(self.last_action),
            "event_history": copy.deepcopy(self.event_history),
            "random_state": self.random.getstate(),
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]):
        self.board = copy.deepcopy(snapshot["board"])
        self.players = copy.deepcopy(snapshot["players"])
        self.current_player_index = snapshot["current_player_index"]
        self.dice = snapshot["dice"]
        self.last_roll_total = snapshot["last_roll_total"]
        self.consecutive_doubles = snapshot["consecutive_doubles"]
        self.extra_turn_pending = snapshot["extra_turn_pending"]
        self.phase = snapshot["phase"]
        self.trade_draft = copy.deepcopy(snapshot.get("trade_draft"))
        self.pending_trade = copy.deepcopy(snapshot["pending_trade"])
        self.last_trade_result = copy.deepcopy(snapshot.get("last_trade_result"))
        self.auction = copy.deepcopy(snapshot["auction"])
        self.trade_proposed_this_turn = snapshot["trade_proposed_this_turn"]
        self.properties_sold_buildings_this_turn = copy.deepcopy(snapshot["properties_sold_buildings_this_turn"])
        self.properties_built_this_turn = copy.deepcopy(snapshot["properties_built_this_turn"])
        self.bank_houses = snapshot["bank_houses"]
        self.bank_hotels = snapshot["bank_hotels"]
        self.chance_deck = copy.deepcopy(snapshot["chance_deck"])
        self.community_deck = copy.deepcopy(snapshot["community_deck"])
        self.done = snapshot["done"]
        self.winner = snapshot["winner"]
        self.turn_count = snapshot["turn_count"]
        self.action_count = snapshot["action_count"]
        self.last_message = snapshot["last_message"]
        self.last_action = copy.deepcopy(snapshot["last_action"])
        self.event_history = copy.deepcopy(snapshot["event_history"])
        self.random.setstate(snapshot["random_state"])

    # ========================================================
    # Main turn actions
    # ========================================================

    def _roll_dice(self):
        if self.phase != "ready_to_roll":
            self.last_message = "Nao e possivel rolar os dados agora."
            return

        player = self.current_player

        if player.bankrupt:
            self._next_turn()
            return

        d1 = self.random.randint(1, 6)
        d2 = self.random.randint(1, 6)
        total = d1 + d2
        is_double = d1 == d2
        self.dice = (d1, d2)
        self.last_roll_total = total

        if player.in_jail:
            self._roll_from_jail(player, total, is_double)
            return

        if is_double:
            self.consecutive_doubles += 1
        else:
            self.consecutive_doubles = 0

        if self.consecutive_doubles >= 3:
            self._send_to_jail(self.current_player_index)
            self.last_message = f"{player.name} tirou tres duplas seguidas e foi para a cadeia."
            self._next_turn()
            return

        self.extra_turn_pending = is_double
        self._move_current_player(total)
        self._resolve_landing()

    def _roll_from_jail(self, player: Player, total: int, is_double: bool):
        if is_double:
            player.in_jail = False
            player.jail_turns = 0
            self.consecutive_doubles = 0
            self.extra_turn_pending = False
            self._move_current_player(total)
            self._resolve_landing()
            return

        player.jail_turns += 1

        if player.jail_turns >= 3:
            player.money -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0
            self._settle_debt(self.current_player_index, None)
            if not player.bankrupt:
                self._move_current_player(total)
                self._resolve_landing()
            return

        self.last_message = f"{player.name} nao tirou dupla para sair da cadeia."
        self._next_turn()

    def _pay_jail_fine(self):
        player = self.current_player
        if not player.in_jail:
            self.last_message = f"{player.name} nao esta na cadeia."
            return

        player.money -= JAIL_FINE
        player.in_jail = False
        player.jail_turns = 0
        self._settle_debt(self.current_player_index, None)
        self.last_message = f"{player.name} pagou ${JAIL_FINE} para sair da cadeia."

    def _use_jail_card(self):
        player = self.current_player
        if not player.in_jail or player.jail_free_cards <= 0:
            self.last_message = "Nao ha carta de saida da cadeia para usar."
            return

        player.jail_free_cards -= 1
        player.in_jail = False
        player.jail_turns = 0
        self.last_message = f"{player.name} usou uma carta de saida da cadeia."

    def _move_current_player(self, steps: int):
        player = self.current_player
        passed_go = player.move(steps, NUM_SPACES)
        if passed_go:
            player.money += PASS_START_BONUS

    def _resolve_landing(self):
        player = self.current_player
        space = self.board[player.position]

        if space.type == "go":
            self.last_message = f"{player.name} caiu no Go."
            self._finish_roll_turn()
            return

        if space.is_ownable():
            self._resolve_ownable_landing(space)
            return

        if space.type == "tax":
            player.money -= space.tax_amount
            self.last_message = f"{player.name} pagou imposto de ${space.tax_amount}."
            self._settle_debt(self.current_player_index, None)
            self._finish_roll_turn()
            return

        if space.type == "chance":
            self._draw_card("chance")
            return

        if space.type == "community_chest":
            self._draw_card("community")
            return

        if space.type == "go_to_jail":
            self._send_to_jail(self.current_player_index)
            self.last_message = f"{player.name} foi enviado para a cadeia."
            self._next_turn()
            return

        if space.type == "jail":
            self.last_message = f"{player.name} esta apenas visitando a cadeia."
            self._finish_roll_turn()
            return

        self.last_message = f"{player.name} caiu em {space.name}."
        self._finish_roll_turn()

    def _resolve_ownable_landing(self, space: Space):
        player = self.current_player

        if space.owner is None:
            self.phase = "awaiting_buy"
            self.last_message = f"{player.name} caiu em {space.name}. Preco: ${space.price}."
            return

        if space.owner == self.current_player_index:
            self.last_message = f"{player.name} caiu na propria propriedade: {space.name}."
            self._finish_roll_turn()
            return

        if space.mortgaged:
            self.last_message = f"{player.name} caiu em {space.name}, mas ela esta hipotecada."
            self._finish_roll_turn()
            return

        rent = self.get_rent_for_space(space)
        owner = self.players[space.owner]
        player.money -= rent
        owner.money += rent
        self.last_message = f"{player.name} pagou ${rent} de aluguel para {owner.name} por {space.name}."
        self._settle_debt(self.current_player_index, space.owner)
        self._finish_roll_turn()

    def _buy_property(self):
        if self.phase != "awaiting_buy":
            self.last_message = "Nao existe propriedade para comprar agora."
            return

        player = self.current_player
        space = self.board[player.position]

        if not space.is_ownable() or space.owner is not None:
            self.phase = "ready_to_roll"
            self._finish_roll_turn()
            return

        if player.money >= space.price:
            player.money -= space.price
            space.owner = self.current_player_index
            self.last_message = f"{player.name} comprou {space.name} por ${space.price}."
        else:
            self.last_message = f"{player.name} nao tinha dinheiro para comprar {space.name}; leilao iniciado."
            self._start_auction(space_index=player.position)
            return

        self.phase = "ready_to_roll"
        self._finish_roll_turn()

    def _pass_buy(self):
        if self.phase != "awaiting_buy":
            self.last_message = "Nao existe compra para passar agora."
            return

        space = self.board[self.current_player.position]
        self.last_message = f"{self.current_player.name} recusou comprar {space.name}; leilao iniciado."
        self._start_auction(space_index=self.current_player.position)

    def _finish_roll_turn(self):
        if self.done or self.phase == "auction":
            return

        self.phase = "ready_to_roll"

        if self.extra_turn_pending and not self.current_player.in_jail and not self.current_player.bankrupt:
            self.extra_turn_pending = False
            self.last_message += " Tirou dupla e joga novamente."
            return

        self._next_turn()

    # ========================================================
    # Auctions
    # ========================================================

    def _start_auction(self, space_index: int):
        active = [
            i
            for i, player in enumerate(self.players)
            if not player.bankrupt and player.money > 0
        ]

        if not active:
            self.phase = "ready_to_roll"
            self._finish_roll_turn()
            return

        self.phase = "auction"
        self.auction = {
            "property_index": space_index,
            "active_bidders": active,
            "current_bid": 0,
            "highest_bidder": None,
            "current_bidder": active[0],
        }

    def _get_auction_actions(self, player_index: int) -> List[Dict[str, Any]]:
        assert self.auction is not None
        player = self.players[player_index]
        current_bid = int(self.auction["current_bid"])
        min_bid = current_bid + 10
        actions = []

        max_bid = self._max_reasonable_auction_bid(player_index)
        if min_bid <= max_bid:
            bid_candidates = sorted(
                {
                    min_bid,
                    min(max_bid, current_bid + 25),
                    min(max_bid, current_bid + 50),
                    min(max_bid, current_bid + 100),
                }
            )
            actions.extend(
                {"type": AUCTION_BID, "amount": amount}
                for amount in bid_candidates
                if current_bid < amount <= max_bid
            )

        if not actions or not self._should_force_auction_bid(player_index, min_bid, max_bid):
            actions.insert(0, {"type": AUCTION_PASS})

        return actions

    def _max_reasonable_auction_bid(self, player_index: int) -> int:
        assert self.auction is not None
        return self._max_reasonable_bid_for_property(player_index, self.auction["property_index"])

    def _max_reasonable_bid_for_property(self, player_index: int, prop_index: int) -> int:
        player = self.players[player_index]
        space = self.board[prop_index]
        if player.money <= 0:
            return 0

        reserve = self._auction_cash_reserve(player_index)
        spendable_cash = max(0, player.money - reserve)
        strategic_value = self._auction_property_value(player_index, space)

        return max(0, min(spendable_cash, strategic_value))

    def _auction_cash_reserve(self, player_index: int) -> int:
        player = self.players[player_index]
        base_reserve = 100

        if player.in_jail:
            base_reserve += JAIL_FINE

        owned_buildings = self._total_houses(player_index)
        if owned_buildings:
            base_reserve += 50

        return min(player.money, base_reserve)

    def _auction_property_value(self, player_index: int, space: Space) -> int:
        if not space.is_ownable():
            return 0

        value = space.price

        if space.type == "property" and space.group in GROUPS:
            owned_in_group = sum(
                1
                for prop_index in GROUPS[space.group]
                if self.board[prop_index].owner == player_index
            )
            opponents_near_monopoly = any(
                sum(1 for prop_index in GROUPS[space.group] if self.board[prop_index].owner == other_index)
                == len(GROUPS[space.group]) - 1
                for other_index in range(len(self.players))
                if other_index != player_index and not self.players[other_index].bankrupt
            )

            if owned_in_group == len(GROUPS[space.group]) - 1:
                value = int(space.price * AUCTION_COMPLETE_VALUE_MULTIPLIER)
            elif owned_in_group > 0:
                value = int(space.price * 1.20)

            if opponents_near_monopoly:
                value = max(value, int(space.price * AUCTION_BLOCK_VALUE_MULTIPLIER))

        elif space.type == "railroad":
            owned_railroads = sum(1 for prop_index in RAILROADS if self.board[prop_index].owner == player_index)
            value = int(space.price * (1.00 + 0.20 * owned_railroads))

        elif space.type == "utility":
            owned_utilities = sum(1 for prop_index in UTILITIES if self.board[prop_index].owner == player_index)
            value = int(space.price * (1.00 + 0.25 * owned_utilities))

        return value

    def _should_force_buy_property(self, player_index: int) -> bool:
        if self.phase != "awaiting_buy":
            return False

        player = self.players[player_index]
        space_index = player.position
        space = self.board[space_index]

        if not space.is_ownable() or space.owner is not None or player.money < space.price:
            return False

        if not self._is_critical_purchase_property(space_index, player_index):
            return False

        return player.money >= space.price + self._critical_purchase_cash_reserve(player_index)

    def _should_force_auction_bid(self, player_index: int, min_bid: int, max_bid: int) -> bool:
        if self.auction is None or min_bid > max_bid:
            return False

        if self.auction.get("highest_bidder") == player_index:
            return False

        prop_index = self.auction["property_index"]
        if not self._is_critical_purchase_property(prop_index, player_index):
            return False

        return self.players[player_index].money >= min_bid + self._critical_purchase_cash_reserve(player_index)

    def _is_critical_purchase_property(self, prop_index: int, player_index: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False

        space = self.board[prop_index]
        if space.type != "property" or space.group not in GROUPS:
            return False

        if self._property_completes_group_for_player(prop_index, player_index):
            return True

        return self._property_blocks_opponent_completion(prop_index, player_index)

    def _property_blocks_opponent_completion(self, prop_index: int, player_index: int) -> bool:
        return any(
            opponent_index != player_index
            and not opponent.bankrupt
            and self._property_completes_group_for_player(prop_index, opponent_index)
            for opponent_index, opponent in enumerate(self.players)
        )

    def _critical_purchase_cash_reserve(self, player_index: int) -> int:
        reserve = CRITICAL_BUY_CASH_RESERVE
        if self.players[player_index].in_jail:
            reserve += JAIL_FINE
        return reserve

    def _auction_bid(self, action: Dict[str, Any]):
        if self.phase != "auction" or not self.auction:
            self.last_message = "Nao ha leilao em andamento."
            return

        bidder_index = self.auction["current_bidder"]
        amount = int(action.get("amount", 0))
        bidder = self.players[bidder_index]

        if amount <= self.auction["current_bid"] or amount > bidder.money:
            self.last_message = f"Lance invalido de {bidder.name}."
            return

        self.auction["current_bid"] = amount
        self.auction["highest_bidder"] = bidder_index
        self.last_message = f"{bidder.name} deu lance de ${amount}."
        self._advance_auction_bidder()

    def _auction_pass(self):
        if self.phase != "auction" or not self.auction:
            self.last_message = "Nao ha leilao em andamento."
            return

        bidder_index = self.auction["current_bidder"]
        bidder = self.players[bidder_index]
        self.auction["active_bidders"] = [
            i for i in self.auction["active_bidders"] if i != bidder_index
        ]
        self.last_message = f"{bidder.name} saiu do leilao."
        self._advance_auction_bidder()

    def _advance_auction_bidder(self):
        assert self.auction is not None
        active = self.auction["active_bidders"]
        highest_bidder = self.auction["highest_bidder"]

        if highest_bidder is not None and active == [highest_bidder]:
            self._finish_auction()
            return

        if not active:
            self._finish_auction()
            return

        current = self.auction["current_bidder"]
        for offset in range(1, len(self.players) + 1):
            candidate = (current + offset) % len(self.players)
            if candidate in active:
                self.auction["current_bidder"] = candidate
                return

    def _recover_auction_state(self) -> bool:
        if self.auction is None:
            self.phase = "ready_to_roll"
            return True

        active = [
            player_index
            for player_index in self.auction["active_bidders"]
            if not self.players[player_index].bankrupt and self.players[player_index].money > 0
        ]
        self.auction["active_bidders"] = active

        highest_bidder = self.auction["highest_bidder"]
        if highest_bidder is not None and highest_bidder not in active:
            self.auction["highest_bidder"] = None
            self.auction["current_bid"] = 0
            highest_bidder = None

        if highest_bidder is not None and active == [highest_bidder]:
            self._finish_auction()
            return True

        if not active:
            self._finish_auction()
            return True

        if self.auction["current_bidder"] not in active:
            self.auction["current_bidder"] = active[0]
            return True

        return False

    def _finish_auction(self):
        assert self.auction is not None
        space = self.board[self.auction["property_index"]]
        highest_bidder = self.auction["highest_bidder"]
        current_bid = int(self.auction["current_bid"])

        if highest_bidder is not None:
            winner = self.players[highest_bidder]
            winner.money -= current_bid
            space.owner = highest_bidder
            self.last_message = f"{winner.name} venceu o leilao de {space.name} por ${current_bid}."
        else:
            self.last_message = f"Ninguem comprou {space.name} no leilao."

        self.auction = None
        self.phase = "ready_to_roll"
        self._finish_roll_turn()

    # ========================================================
    # Buildings, mortgages and trades
    # ========================================================

    def _get_financial_actions(self, player_index: int) -> List[Dict[str, Any]]:
        actions = []
        actions.extend(self._get_build_actions(player_index))
        if self._needs_defensive_cash(player_index):
            actions.extend(self._get_sell_house_actions(player_index))
            actions.extend(self._get_mortgage_actions(player_index))
        actions.extend(self._get_unmortgage_actions(player_index))
        return actions

    def _needs_defensive_cash(self, player_index: int) -> bool:
        player = self.players[player_index]
        return player.money < DEFENSIVE_CASH_THRESHOLD

    def _build_house(self, action: Dict[str, Any]):
        prop_index = int(action.get("property_index", -1))
        if prop_index < 0 or prop_index >= len(self.board):
            self.last_message = "Propriedade invalida para construcao."
            return

        if not self._can_build_on_property(self.current_player_index, prop_index):
            self.last_message = "Construcao nao permitida nessa propriedade."
            return

        player = self.current_player
        space = self.board[prop_index]
        player.money -= space.build_cost
        self.properties_built_this_turn.add(prop_index)

        if space.houses == 4:
            self.bank_hotels -= 1
            self.bank_houses += 4
        else:
            self.bank_houses -= 1

        space.houses += 1
        label = "hotel" if space.houses == MAX_HOUSES else f"{space.houses} casa(s)"
        self.last_message = f"{player.name} construiu {label} em {space.name}."

    def _sell_house(self, action: Dict[str, Any]):
        prop_index = int(action.get("property_index", -1))
        if prop_index < 0 or prop_index >= len(self.board):
            self.last_message = "Propriedade invalida para venda."
            return

        space = self.board[prop_index]
        if not self._can_sell_building(self.current_player_index, prop_index):
            self.last_message = "Venda de construcao nao permitida nessa propriedade."
            return

        player = self.current_player
        player.money += space.build_cost // 2

        if space.houses == MAX_HOUSES:
            self.bank_hotels += 1
            self.bank_houses -= 4
        else:
            self.bank_houses += 1

        space.houses -= 1
        self.properties_sold_buildings_this_turn.add(prop_index)
        self.last_message = f"{player.name} vendeu uma construcao em {space.name}."

    def _mortgage_property(self, action: Dict[str, Any]):
        prop_index = int(action.get("property_index", -1))
        if prop_index < 0 or prop_index >= len(self.board):
            self.last_message = "Propriedade invalida para hipoteca."
            return

        space = self.board[prop_index]
        if not self._can_mortgage_property(self.current_player_index, prop_index):
            self.last_message = "Hipoteca nao permitida nessa propriedade."
            return

        space.mortgaged = True
        self.current_player.money += space.mortgage_value
        self.last_message = f"{self.current_player.name} hipotecou {space.name} por ${space.mortgage_value}."

    def _unmortgage_property(self, action: Dict[str, Any]):
        prop_index = int(action.get("property_index", -1))
        if prop_index < 0 or prop_index >= len(self.board):
            self.last_message = "Propriedade invalida para quitar hipoteca."
            return

        space = self.board[prop_index]
        cost = self._unmortgage_cost(space)

        if space.owner != self.current_player_index or not space.mortgaged or self.current_player.money < cost:
            self.last_message = "Nao e possivel quitar essa hipoteca."
            return

        self.current_player.money -= cost
        space.mortgaged = False
        self.last_message = f"{self.current_player.name} quitou hipoteca de {space.name} por ${cost}."

    def _can_start_trade(self, player_index: int) -> bool:
        if self.trade_proposed_this_turn or self.players[player_index].bankrupt:
            return False

        proposer_can_give = bool(self.get_tradeable_properties(player_index)) or self.players[player_index].money > 0
        if not proposer_can_give:
            return False

        for target_index, target in enumerate(self.players):
            if target_index == player_index or target.bankrupt:
                continue
            if self.get_tradeable_properties(target_index) or target.money > 0:
                return True
        return False

    def _start_trade(self):
        if self.phase != "ready_to_roll":
            self.last_message = "Trocas so podem ser iniciadas antes de rolar os dados."
            return

        proposer_index = self.current_player_index
        if not self._can_start_trade(proposer_index):
            self.last_message = "Nao ha troca legal disponivel para iniciar."
            return

        self.trade_draft = {
            "from": proposer_index,
            "to": None,
            "offer_properties": [],
            "request_properties": [],
            "offer_money": 0,
            "request_money": 0,
            "stage": "target",
        }
        self.trade_proposed_this_turn = True
        self.phase = "building_trade"
        self.last_trade_result = None
        self.last_message = f"{self.current_player.name} iniciou uma proposta de troca."

    def _get_trade_draft_actions(self, player_index: int) -> List[Dict[str, Any]]:
        if not self.trade_draft or self.trade_draft["from"] != player_index:
            return []

        actions: List[Dict[str, Any]] = [{"type": CANCEL_TRADE}]
        stage = self.trade_draft.get("stage", "target")
        target_index = self.trade_draft.get("to")

        if stage == "target":
            for candidate_index, candidate in enumerate(self.players):
                if candidate_index == player_index or candidate.bankrupt:
                    continue
                if self.get_tradeable_properties(candidate_index) or candidate.money > 0:
                    actions.append({"type": SELECT_TRADE_TARGET, "target_player": candidate_index})
            return actions

        if target_index is None or not (0 <= target_index < self.num_players) or self.players[target_index].bankrupt:
            return actions

        if stage == "offer_properties":
            selected = set(self.trade_draft["offer_properties"])
            if len(selected) < TRADE_MAX_PROPERTIES_PER_SIDE:
                for prop_index in self.get_tradeable_properties(player_index):
                    if prop_index not in selected:
                        actions.append({"type": ADD_TRADE_OFFER_PROPERTY, "property_index": prop_index})
            actions.append({"type": FINISH_TRADE_OFFER})
            return actions

        if stage == "offer_money":
            for amount in self._trade_money_options(self.players[player_index].money):
                actions.append({
                    "type": SET_TRADE_OFFER_MONEY,
                    "amount": amount,
                    "offer_money": amount,
                })
            return actions

        if stage == "request_properties":
            selected = set(self.trade_draft["request_properties"])
            if len(selected) < TRADE_MAX_PROPERTIES_PER_SIDE:
                for prop_index in self.get_tradeable_properties(target_index):
                    if prop_index not in selected:
                        actions.append({"type": ADD_TRADE_REQUEST_PROPERTY, "property_index": prop_index})
            actions.append({"type": FINISH_TRADE_REQUEST})
            return actions

        if stage == "request_money":
            for amount in self._trade_money_options(self.players[target_index].money):
                actions.append({
                    "type": SET_TRADE_REQUEST_MONEY,
                    "amount": amount,
                    "request_money": amount,
                })
            return actions

        if stage == "confirm":
            if self._trade_draft_is_legal():
                actions.append(self._draft_action_payload(SUBMIT_TRADE))
            return actions

        return actions

    def _trade_money_options(self, max_amount: int) -> List[int]:
        max_amount = max(0, int(max_amount))
        values = {0}
        values.update(amount for amount in TRADE_MONEY_CANDIDATES if amount <= max_amount)
        if max_amount > 0:
            values.add(max_amount)
        return sorted(values)

    def _draft_action_payload(self, action_type: str) -> Dict[str, Any]:
        draft = self.trade_draft or {}
        return {
            "type": action_type,
            "target_player": draft.get("to", -1),
            "offer_properties": list(draft.get("offer_properties", [])),
            "offer_money": int(draft.get("offer_money", 0)),
            "request_properties": list(draft.get("request_properties", [])),
            "request_money": int(draft.get("request_money", 0)),
        }

    def _require_trade_draft_stage(self, stage: str) -> bool:
        if self.phase != "building_trade" or not self.trade_draft:
            self.last_message = "Nao existe proposta de troca em montagem."
            return False
        if self.trade_draft.get("stage") != stage:
            self.last_message = "Essa acao nao pertence a etapa atual da troca."
            return False
        return True

    def _select_trade_target(self, action: Dict[str, Any]):
        if not self._require_trade_draft_stage("target"):
            return

        target_index = action.get("target_player")
        proposer_index = self.trade_draft["from"]
        if target_index is None or target_index == proposer_index or not (0 <= target_index < self.num_players):
            self.last_message = "Jogador alvo invalido para troca."
            return
        if self.players[target_index].bankrupt:
            self.last_message = "Jogador falido nao pode participar de troca."
            return
        if not self.get_tradeable_properties(target_index) and self.players[target_index].money <= 0:
            self.last_message = "Jogador alvo nao tem itens ou dinheiro negociaveis."
            return

        self.trade_draft["to"] = int(target_index)
        self.trade_draft["stage"] = "offer_properties"
        self.last_message = f"{self.players[proposer_index].name} escolheu {self.players[target_index].name} para troca."

    def _add_trade_offer_property(self, action: Dict[str, Any]):
        if not self._require_trade_draft_stage("offer_properties"):
            return

        proposer_index = self.trade_draft["from"]
        prop_index = int(action.get("property_index", -1))
        selected = self.trade_draft["offer_properties"]
        if len(selected) >= TRADE_MAX_PROPERTIES_PER_SIDE:
            self.last_message = "Limite de propriedades oferecidas atingido."
            return
        if prop_index in selected:
            self.last_message = "Propriedade ja esta na oferta."
            return
        if not self._is_tradeable_property(prop_index, proposer_index):
            self.last_message = "Propriedade oferecida nao pode ser negociada."
            return

        selected.append(prop_index)
        self.last_message = f"{self.current_player.name} adicionou {self.board[prop_index].name} a oferta."

    def _finish_trade_offer(self):
        if not self._require_trade_draft_stage("offer_properties"):
            return
        self.trade_draft["stage"] = "offer_money"
        self.last_message = "Etapa de propriedades oferecidas finalizada."

    def _set_trade_offer_money(self, action: Dict[str, Any]):
        if not self._require_trade_draft_stage("offer_money"):
            return

        amount = int(action.get("amount", action.get("offer_money", 0)))
        proposer_index = self.trade_draft["from"]
        if amount < 0 or amount > self.players[proposer_index].money:
            self.last_message = "Dinheiro oferecido invalido."
            return

        self.trade_draft["offer_money"] = amount
        self.trade_draft["stage"] = "request_properties"
        self.last_message = f"{self.current_player.name} definiu oferta em dinheiro: ${amount}."

    def _add_trade_request_property(self, action: Dict[str, Any]):
        if not self._require_trade_draft_stage("request_properties"):
            return

        target_index = self.trade_draft.get("to")
        prop_index = int(action.get("property_index", -1))
        selected = self.trade_draft["request_properties"]
        if target_index is None:
            self.last_message = "Troca ainda nao tem jogador alvo."
            return
        if len(selected) >= TRADE_MAX_PROPERTIES_PER_SIDE:
            self.last_message = "Limite de propriedades pedidas atingido."
            return
        if prop_index in selected:
            self.last_message = "Propriedade ja esta no pedido."
            return
        if not self._is_tradeable_property(prop_index, int(target_index)):
            self.last_message = "Propriedade pedida nao pode ser negociada."
            return

        selected.append(prop_index)
        self.last_message = f"{self.current_player.name} pediu {self.board[prop_index].name}."

    def _finish_trade_request(self):
        if not self._require_trade_draft_stage("request_properties"):
            return
        self.trade_draft["stage"] = "request_money"
        self.last_message = "Etapa de propriedades pedidas finalizada."

    def _set_trade_request_money(self, action: Dict[str, Any]):
        if not self._require_trade_draft_stage("request_money"):
            return

        target_index = self.trade_draft.get("to")
        amount = int(action.get("amount", action.get("request_money", 0)))
        if target_index is None or amount < 0 or amount > self.players[int(target_index)].money:
            self.last_message = "Dinheiro pedido invalido."
            return

        self.trade_draft["request_money"] = amount
        self.trade_draft["stage"] = "confirm"
        self.last_message = f"{self.current_player.name} definiu pedido em dinheiro: ${amount}."

    def _trade_draft_is_legal(self) -> bool:
        if not self.trade_draft:
            return False

        proposer_index = self.trade_draft["from"]
        target_index = self.trade_draft.get("to")
        if target_index is None or not (0 <= target_index < self.num_players):
            return False

        proposer = self.players[proposer_index]
        target = self.players[int(target_index)]
        if proposer.bankrupt or target.bankrupt:
            return False

        offer_properties = list(self.trade_draft.get("offer_properties", []))
        request_properties = list(self.trade_draft.get("request_properties", []))
        offer_money = int(self.trade_draft.get("offer_money", 0))
        request_money = int(self.trade_draft.get("request_money", 0))

        if offer_money < 0 or request_money < 0:
            return False
        if offer_money > proposer.money or request_money > target.money:
            return False
        if not self._has_meaningful_trade_consideration(
            offer_properties,
            offer_money,
            request_properties,
            request_money,
        ):
            return False

        for prop_index in offer_properties:
            if not self._is_tradeable_property(prop_index, proposer_index):
                return False
        for prop_index in request_properties:
            if not self._is_tradeable_property(prop_index, int(target_index)):
                return False
        return True

    def _submit_trade(self):
        if not self._require_trade_draft_stage("confirm"):
            return
        if not self._trade_draft_is_legal():
            self.last_message = "Proposta de troca invalida ou sem contrapartida."
            return

        draft = self.trade_draft
        proposer = self.players[draft["from"]]
        target = self.players[draft["to"]]
        self.pending_trade = {
            "from": draft["from"],
            "to": draft["to"],
            "offer_properties": list(draft["offer_properties"]),
            "request_properties": list(draft["request_properties"]),
            "offer_money": int(draft["offer_money"]),
            "request_money": int(draft["request_money"]),
        }
        self.trade_draft = None
        self.last_trade_result = None
        self.phase = "pending_trade_response"
        self.last_message = f"{proposer.name} enviou uma proposta de troca para {target.name}."

    def _cancel_trade(self):
        if self.phase != "building_trade" or not self.trade_draft:
            self.last_message = "Nao existe proposta de troca para cancelar."
            return
        proposer = self.players[self.trade_draft["from"]]
        self._clear_trade_draft()
        self.last_message = f"{proposer.name} cancelou a montagem da troca."

    def _clear_trade_draft(self):
        self.trade_draft = None
        self.phase = "ready_to_roll"

    def _propose_trade(self, action: Dict[str, Any]):
        if self.phase != "ready_to_roll":
            self.last_message = "Trocas so podem ser propostas antes de rolar os dados."
            return

        if self.trade_proposed_this_turn:
            self.last_message = "Esse jogador ja propos uma troca neste turno."
            return

        proposer_index = self.current_player_index
        target_index = action.get("target_player")

        if target_index is None or target_index == proposer_index or not (0 <= target_index < len(self.players)):
            self.last_message = "Jogador alvo invalido para troca."
            return

        proposer = self.players[proposer_index]
        target = self.players[target_index]

        if proposer.bankrupt or target.bankrupt:
            self.last_message = "Jogador falido nao pode participar de troca."
            return

        offer_properties = list(action.get("offer_properties", []))
        request_properties = list(action.get("request_properties", []))
        offer_money = int(action.get("offer_money", 0))
        request_money = int(action.get("request_money", 0))

        if offer_money < 0 or request_money < 0:
            self.last_message = "Valores de dinheiro na troca nao podem ser negativos."
            return

        if not offer_properties and not request_properties and offer_money == 0 and request_money == 0:
            self.last_message = "Troca vazia ignorada."
            return

        if offer_money > proposer.money or request_money > target.money:
            self.last_message = "Troca cancelada por dinheiro insuficiente."
            return

        for prop_index in offer_properties:
            if not self._is_tradeable_property(prop_index, proposer_index):
                self.last_message = "Uma propriedade oferecida nao pode ser negociada."
                return

        for prop_index in request_properties:
            if not self._is_tradeable_property(prop_index, target_index):
                self.last_message = "Uma propriedade pedida nao pode ser negociada."
                return

        if not self._has_meaningful_trade_consideration(
            offer_properties,
            offer_money,
            request_properties,
            request_money,
        ):
            self.last_message = "Troca ignorada por falta de contrapartida real."
            return

        self.pending_trade = {
            "from": proposer_index,
            "to": target_index,
            "offer_properties": offer_properties,
            "request_properties": request_properties,
            "offer_money": offer_money,
            "request_money": request_money,
        }
        self.last_trade_result = None
        self.phase = "pending_trade_response"
        self.trade_proposed_this_turn = True
        self.last_message = f"{proposer.name} propos uma troca para {target.name}."

    def _accept_trade(self):
        if self.phase != "pending_trade_response" or self.pending_trade is None:
            self.last_message = "Nao existe troca pendente."
            return

        trade = self.pending_trade
        proposer_index = trade["from"]
        target_index = trade["to"]
        proposer = self.players[proposer_index]
        target = self.players[target_index]

        if trade["offer_money"] > proposer.money or trade["request_money"] > target.money:
            self.last_message = "Troca cancelada por dinheiro insuficiente."
            self._clear_trade()
            return

        for prop_index in trade["offer_properties"]:
            if not self._is_tradeable_property(prop_index, proposer_index):
                self.last_message = "Troca cancelada: propriedade oferecida nao pode ser negociada."
                self._clear_trade()
                return

        for prop_index in trade["request_properties"]:
            if not self._is_tradeable_property(prop_index, target_index):
                self.last_message = "Troca cancelada: propriedade pedida nao pode ser negociada."
                self._clear_trade()
                return

        proposer.money -= trade["offer_money"]
        target.money += trade["offer_money"]
        target.money -= trade["request_money"]
        proposer.money += trade["request_money"]

        for prop_index in trade["offer_properties"]:
            self.board[prop_index].owner = target_index
            self._charge_mortgage_transfer_interest(target_index, prop_index)

        for prop_index in trade["request_properties"]:
            self.board[prop_index].owner = proposer_index
            self._charge_mortgage_transfer_interest(proposer_index, prop_index)

        self.last_message = f"{target.name} aceitou a troca de {proposer.name}."
        self._set_last_trade_result("accepted", trade, self.last_message)
        self._clear_trade()

    def _decline_trade(self):
        if self.phase != "pending_trade_response" or self.pending_trade is None:
            self.last_message = "Nao existe troca pendente."
            return

        proposer = self.players[self.pending_trade["from"]]
        target = self.players[self.pending_trade["to"]]
        self.last_message = f"{target.name} recusou a troca de {proposer.name}."
        self._set_last_trade_result("declined", self.pending_trade, self.last_message)
        self._clear_trade()

    def _clear_trade(self):
        self.pending_trade = None
        self.trade_draft = None
        self.phase = "ready_to_roll"

    def _set_last_trade_result(self, status: str, trade: Dict[str, Any], message: str):
        self.last_trade_result = copy.deepcopy(trade)
        self.last_trade_result["status"] = status
        self.last_trade_result["message"] = message

    # ========================================================
    # Cards
    # ========================================================

    def _make_chance_deck(self) -> List[Dict[str, Any]]:
        deck = [
            {"type": "move_to", "space": GO_INDEX, "message": "avancou para Go"},
            {"type": "move_to", "space": 24, "message": "avancou para Red 3"},
            {"type": "move_to", "space": 11, "message": "avancou para Pink 1"},
            {"type": "nearest_utility"},
            {"type": "nearest_railroad", "double_rent": True},
            {"type": "nearest_railroad", "double_rent": True},
            {"type": "money", "amount": 50, "message": "recebeu dividendo"},
            {"type": "jail_card"},
            {"type": "move_relative", "spaces": -3, "message": "voltou 3 casas"},
            {"type": "go_to_jail"},
            {"type": "repairs", "house": 25, "hotel": 100},
            {"type": "money", "amount": -15, "message": "pagou taxa"},
            {"type": "move_to", "space": 5, "message": "avancou para Railroad 1"},
            {"type": "move_to", "space": 39, "message": "avancou para Dark Blue 2"},
            {"type": "pay_each", "amount": 50},
            {"type": "money", "amount": 150, "message": "recebeu premio"},
        ]
        self.random.shuffle(deck)
        return deck

    def _make_community_deck(self) -> List[Dict[str, Any]]:
        deck = [
            {"type": "move_to", "space": GO_INDEX, "message": "avancou para Go"},
            {"type": "money", "amount": 200, "message": "recebeu pagamento do banco"},
            {"type": "money", "amount": -50, "message": "pagou taxa medica"},
            {"type": "money", "amount": 50, "message": "recebeu venda de acoes"},
            {"type": "jail_card"},
            {"type": "go_to_jail"},
            {"type": "money", "amount": 100, "message": "recebeu presente"},
            {"type": "money", "amount": 20, "message": "recebeu reembolso"},
            {"type": "money", "amount": 100, "message": "recebeu seguro"},
            {"type": "money", "amount": -100, "message": "pagou hospital"},
            {"type": "money", "amount": -50, "message": "pagou escola"},
            {"type": "money", "amount": 25, "message": "recebeu consultoria"},
            {"type": "repairs", "house": 40, "hotel": 115},
            {"type": "money", "amount": 10, "message": "recebeu premio pequeno"},
            {"type": "money", "amount": 100, "message": "recebeu heranca"},
            {"type": "collect_each", "amount": 50},
        ]
        self.random.shuffle(deck)
        return deck

    def _draw_card(self, deck_name: str):
        deck = self.chance_deck if deck_name == "chance" else self.community_deck
        if not deck:
            deck.extend(self._make_chance_deck() if deck_name == "chance" else self._make_community_deck())

        card = deck.pop(0)
        self._apply_card(card)

    def _apply_card(self, card: Dict[str, Any]):
        player = self.current_player
        card_type = card["type"]

        if card_type == "money":
            amount = int(card["amount"])
            player.money += amount
            self.last_message = f"{player.name} {card['message']} (${amount})."
            self._settle_debt(self.current_player_index, None)
            self._finish_roll_turn()
            return

        if card_type == "move_to":
            target = int(card["space"])
            if target < player.position:
                player.money += PASS_START_BONUS
            player.position = target
            self.last_message = f"{player.name} {card['message']}."
            self._resolve_landing()
            return

        if card_type == "move_relative":
            player.position = (player.position + int(card["spaces"])) % NUM_SPACES
            self.last_message = f"{player.name} {card['message']}."
            self._resolve_landing()
            return

        if card_type == "nearest_railroad":
            self._advance_to_nearest(RAILROADS)
            space = self.board[player.position]
            self.last_message = f"{player.name} avancou para {space.name}."
            self._resolve_landing_with_rent_multiplier(2)
            return

        if card_type == "nearest_utility":
            self._advance_to_nearest(UTILITIES)
            space = self.board[player.position]
            self.last_message = f"{player.name} avancou para {space.name}."
            self._resolve_landing_with_rent_multiplier(1, utility_multiplier=10)
            return

        if card_type == "go_to_jail":
            self._send_to_jail(self.current_player_index)
            self.last_message = f"{player.name} comprou uma passagem direta para a cadeia."
            self._next_turn()
            return

        if card_type == "jail_card":
            player.jail_free_cards += 1
            self.last_message = f"{player.name} recebeu carta de saida da cadeia."
            self._finish_roll_turn()
            return

        if card_type == "repairs":
            cost = self._repair_cost(self.current_player_index, int(card["house"]), int(card["hotel"]))
            player.money -= cost
            self.last_message = f"{player.name} pagou ${cost} em reparos."
            self._settle_debt(self.current_player_index, None)
            self._finish_roll_turn()
            return

        if card_type == "pay_each":
            amount = int(card["amount"])
            for i, other in enumerate(self.players):
                if i != self.current_player_index and not other.bankrupt:
                    player.money -= amount
                    other.money += amount
            self.last_message = f"{player.name} pagou ${amount} para cada jogador."
            self._settle_debt(self.current_player_index, None)
            self._finish_roll_turn()
            return

        if card_type == "collect_each":
            amount = int(card["amount"])
            for i, other in enumerate(self.players):
                if i != self.current_player_index and not other.bankrupt:
                    other.money -= amount
                    player.money += amount
                    self._settle_debt(i, self.current_player_index)
            self.last_message = f"{player.name} recebeu ${amount} de cada jogador."
            self._finish_roll_turn()

    def _advance_to_nearest(self, indexes: List[int]):
        player = self.current_player
        candidates = [index for index in indexes if index > player.position]
        target = candidates[0] if candidates else indexes[0]
        if target < player.position:
            player.money += PASS_START_BONUS
        player.position = target

    def _resolve_landing_with_rent_multiplier(self, rent_multiplier: int, utility_multiplier: Optional[int] = None):
        space = self.board[self.current_player.position]
        if not space.is_ownable() or space.owner is None or space.owner == self.current_player_index:
            self._resolve_landing()
            return

        if space.mortgaged:
            self._finish_roll_turn()
            return

        if utility_multiplier is not None:
            rent = self.last_roll_total * utility_multiplier
        else:
            rent = self.get_rent_for_space(space) * rent_multiplier

        player = self.current_player
        owner = self.players[space.owner]
        player.money -= rent
        owner.money += rent
        self.last_message += f" Pagou ${rent} para {owner.name}."
        self._settle_debt(self.current_player_index, space.owner)
        self._finish_roll_turn()

    # ========================================================
    # Rule helpers
    # ========================================================

    def get_rent_for_space(self, space: Space) -> int:
        if not space.is_ownable() or space.mortgaged:
            return 0

        if space.type == "property":
            rent = space.current_rent()
            if space.houses == 0 and space.group and self._owns_complete_group(space.owner, space.group):
                return rent * 2
            return rent

        if space.type == "railroad":
            owned_count = sum(1 for i in RAILROADS if self.board[i].owner == space.owner and not self.board[i].mortgaged)
            return 25 * (2 ** max(0, owned_count - 1))

        if space.type == "utility":
            owned_count = sum(1 for i in UTILITIES if self.board[i].owner == space.owner and not self.board[i].mortgaged)
            multiplier = 10 if owned_count >= 2 else 4
            return self.last_roll_total * multiplier

        return space.rent

    def _send_to_jail(self, player_index: int):
        player = self.players[player_index]
        player.position = JAIL_INDEX
        player.in_jail = True
        player.jail_turns = 0
        self.consecutive_doubles = 0
        self.extra_turn_pending = False
        self.phase = "ready_to_roll"

    def _next_turn(self):
        if self.done:
            return

        self.phase = "ready_to_roll"
        self.trade_draft = None
        self.trade_proposed_this_turn = False
        self.properties_sold_buildings_this_turn.clear()
        self.properties_built_this_turn.clear()
        self.extra_turn_pending = False
        self.consecutive_doubles = 0
        self.turn_count += 1

        if self.turn_count >= MAX_TURNS:
            self._finish_by_net_worth()
            return

        for _ in range(len(self.players)):
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            if not self.current_player.bankrupt:
                return

    def _settle_debt(self, player_index: int, creditor_index: Optional[int]):
        player = self.players[player_index]
        if player.money >= 0 or player.bankrupt:
            return

        self._auto_sell_buildings(player_index)
        self._auto_mortgage_properties(player_index)

        if player.money < 0:
            self._declare_bankruptcy(player_index, creditor_index)

    def _auto_sell_buildings(self, player_index: int):
        while self.players[player_index].money < 0:
            sellable = self._get_sell_house_actions(player_index)
            if not sellable:
                return
            action = max(
                sellable,
                key=lambda a: self.board[a["property_index"]].build_cost,
            )
            prop_index = action["property_index"]
            space = self.board[prop_index]
            self.players[player_index].money += space.build_cost // 2
            if space.houses == MAX_HOUSES:
                self.bank_hotels += 1
                self.bank_houses -= 4
            else:
                self.bank_houses += 1
            space.houses -= 1

    def _auto_mortgage_properties(self, player_index: int):
        while self.players[player_index].money < 0:
            mortgageable = self._get_mortgage_actions(player_index)
            if not mortgageable:
                return
            prop_index = mortgageable[0]["property_index"]
            space = self.board[prop_index]
            space.mortgaged = True
            self.players[player_index].money += space.mortgage_value

    def _declare_bankruptcy(self, player_index: int, creditor_index: Optional[int]):
        player = self.players[player_index]
        player.bankrupt = True

        for space in self.board:
            if space.owner != player_index:
                continue
            space.houses = 0
            space.owner = creditor_index
            if creditor_index is None:
                space.mortgaged = False
            elif space.mortgaged:
                self._charge_mortgage_transfer_interest(creditor_index, self.board.index(space))

        if creditor_index is not None:
            self.players[creditor_index].money += max(0, player.money)

        player.money = 0
        self.last_message += f" {player.name} faliu!"

    def _check_game_end(self):
        active_players = [i for i, p in enumerate(self.players) if not p.bankrupt]
        if len(active_players) == 1:
            self.done = True
            self.phase = "game_over"
            self.winner = active_players[0]
            self.last_message = f"Fim de jogo! {self.players[self.winner].name} venceu!"
        elif len(active_players) == 0:
            self.done = True
            self.phase = "game_over"
            self.winner = max(range(len(self.players)), key=lambda i: self._reward_net_worth(i))
            self.last_message = f"Fim de jogo! {self.players[self.winner].name} venceu por patrimonio residual."

    def _finish_by_net_worth(self):
        self.done = True
        self.phase = "game_over"
        self.winner = max(range(len(self.players)), key=lambda i: self._net_worth(i))
        self.last_message = f"Limite de turnos atingido. {self.players[self.winner].name} venceu por patrimonio."

    def _net_worth(self, player_index: int) -> int:
        player = self.players[player_index]
        if player.bankrupt:
            return -999999

        total = player.money
        for space in self.board:
            if space.owner == player_index:
                total += space.mortgage_value if space.mortgaged else space.price
                total += space.houses * space.build_cost
        return total

    def _reward_net_worth(self, player_index: int) -> int:
        player = self.players[player_index]
        total = player.money
        for space in self.board:
            if space.owner == player_index:
                total += space.mortgage_value if space.mortgaged else space.price
                total += space.houses * space.build_cost
        return total

    def _strategic_reward(
        self,
        action_type: Optional[str],
        before_completed_groups: int,
        after_completed_groups: int,
        before_total_houses: int,
        after_total_houses: int,
    ) -> float:
        reward = 0.0
        if action_type in (BUY_PROPERTY, AUCTION_BID):
            reward += 1.0
        if action_type == BUILD_HOUSE and after_total_houses > before_total_houses:
            reward += 4.0
        completed_delta = after_completed_groups - before_completed_groups
        if completed_delta > 0:
            reward += 12.0 * completed_delta
        return reward

    def _purchase_auction_strategy_reward(
        self,
        action_type: Optional[str],
        acting_player_index: int,
        action: Dict[str, Any],
        purchase_property_index: Optional[int],
        auction_before_action: Optional[Dict[str, Any]],
    ) -> float:
        if action_type == BUY_PROPERTY and purchase_property_index is not None:
            return self._property_control_reward(acting_player_index, purchase_property_index)

        if action_type == PASS_BUY and purchase_property_index is not None:
            if self._can_afford_critical_purchase(acting_player_index, purchase_property_index):
                return -self._property_control_penalty(acting_player_index, purchase_property_index)
            return 0.0

        if action_type == AUCTION_BID and auction_before_action is not None:
            prop_index = int(auction_before_action["property_index"])
            amount = int(action.get("amount", 0))
            max_bid = max(1, self._max_reasonable_bid_for_property(acting_player_index, prop_index))
            bid_pressure = min(1.0, amount / float(max_bid))
            return self._property_control_reward(acting_player_index, prop_index) * (0.35 + 0.35 * bid_pressure)

        if action_type == AUCTION_PASS and auction_before_action is not None:
            prop_index = int(auction_before_action["property_index"])
            current_bid = int(auction_before_action.get("current_bid", 0))
            min_bid = current_bid + 10
            max_bid = self._max_reasonable_bid_for_property(acting_player_index, prop_index)

            penalty = 0.0
            if min_bid <= max_bid and self._is_critical_purchase_property(prop_index, acting_player_index):
                penalty += self._property_control_penalty(acting_player_index, prop_index)

            highest_bidder = auction_before_action.get("highest_bidder")
            if (
                highest_bidder is not None
                and highest_bidder != acting_player_index
                and self._property_completes_group_for_player(prop_index, int(highest_bidder))
            ):
                penalty += 5.0

            return -min(10.0, penalty)

        return 0.0

    def _current_purchase_property_index(self) -> Optional[int]:
        if self.phase != "awaiting_buy":
            return None
        prop_index = self.current_player.position
        if not (0 <= prop_index < len(self.board)):
            return None
        space = self.board[prop_index]
        if not space.is_ownable() or space.owner is not None:
            return None
        return prop_index

    def _property_control_reward(self, player_index: int, prop_index: int) -> float:
        if not (0 <= prop_index < len(self.board)):
            return 0.0

        space = self.board[prop_index]
        if space.type != "property" or space.group not in GROUPS:
            return 1.0 if space.is_ownable() else 0.0

        reward = 1.0
        if self._property_completes_group_for_player(prop_index, player_index):
            reward += 8.0
        elif self._property_creates_near_group_for_player(prop_index, player_index):
            reward += 2.5

        if self._property_blocks_opponent_completion(prop_index, player_index):
            reward += 8.0

        return min(10.0, reward)

    def _property_control_penalty(self, player_index: int, prop_index: int) -> float:
        reward = self._property_control_reward(player_index, prop_index)
        if self._property_blocks_opponent_completion(prop_index, player_index):
            reward = max(reward, 9.0)
        if self._property_completes_group_for_player(prop_index, player_index):
            reward = max(reward, 9.0)
        return min(10.0, reward)

    def _can_afford_critical_purchase(self, player_index: int, prop_index: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False
        space = self.board[prop_index]
        if not space.is_ownable():
            return False
        return self.players[player_index].money >= space.price + self._critical_purchase_cash_reserve(player_index)

    def _completed_group_count(self, player_index: int) -> int:
        return sum(1 for group in GROUPS if self._owns_complete_group(player_index, group))

    def _near_group_count(self, player_index: int) -> int:
        count = 0
        for group, properties in GROUPS.items():
            owned = sum(1 for prop_index in properties if self.board[prop_index].owner == player_index)
            if owned == len(properties) - 1:
                count += 1
        return count

    def _blocking_property_count(self, player_index: int) -> int:
        blockers = 0
        for group, properties in GROUPS.items():
            for prop_index in properties:
                if self.board[prop_index].owner != player_index:
                    continue
                for opponent_index, opponent in enumerate(self.players):
                    if opponent_index == player_index or opponent.bankrupt:
                        continue
                    opponent_owned = sum(
                        1
                        for other_index in properties
                        if self.board[other_index].owner == opponent_index
                    )
                    if opponent_owned == len(properties) - 1:
                        blockers += 1
                        break
        return blockers

    def _highest_rent_potential(self, player_index: int) -> int:
        highest = 0
        for space in self.board:
            if space.owner != player_index:
                continue
            if space.type == "property" and space.rent_schedule:
                highest = max(highest, max(space.rent_schedule))
            elif space.type == "railroad":
                highest = max(highest, 200)
            elif space.type == "utility":
                highest = max(highest, 120)
        return highest

    def _best_group_progress(self, player_index: int) -> float:
        best = 0.0
        for properties in GROUPS.values():
            owned = sum(1 for prop_index in properties if self.board[prop_index].owner == player_index)
            best = max(best, owned / float(len(properties)))
        return best

    def _strategic_position_score(self, player_index: int) -> float:
        return (
            self._reward_net_worth(player_index)
            + self._completed_group_count(player_index) * 650.0
            + self._near_group_count(player_index) * 180.0
            + self._blocking_property_count(player_index) * 90.0
            + self._highest_rent_potential(player_index) * 0.45
            + self._total_houses(player_index) * 80.0
        )

    def _total_houses(self, player_index: int) -> int:
        return sum(
            space.houses
            for space in self.board
            if space.type == "property" and space.owner == player_index
        )

    def _owns_complete_group(self, player_index: Optional[int], group: Optional[str]) -> bool:
        if player_index is None or group is None or group not in GROUPS:
            return False
        return all(self.board[prop_index].owner == player_index for prop_index in GROUPS[group])

    def _is_valid_property_owner(self, prop_index: int, owner_index: int) -> bool:
        return (
            0 <= prop_index < len(self.board)
            and self.board[prop_index].is_ownable()
            and self.board[prop_index].owner == owner_index
        )

    def _is_tradeable_property(self, prop_index: int, owner_index: int) -> bool:
        if not self._is_valid_property_owner(prop_index, owner_index):
            return False

        space = self.board[prop_index]
        if space.type == "property" and self._group_has_buildings(space.group):
            return False

        return True

    def _has_meaningful_trade_consideration(
        self,
        offer_properties: List[int],
        offer_money: int,
        request_properties: List[int],
        request_money: int,
    ) -> bool:
        proposer_gives = bool(offer_properties) or offer_money > 0
        target_gives = bool(request_properties) or request_money > 0
        return proposer_gives and target_gives

    def _estimate_trade_score_deltas(
        self,
        proposer_index: int,
        target_index: int,
        offer_properties: List[int],
        offer_money: int,
        request_properties: List[int],
        request_money: int,
    ) -> tuple[float, float]:
        before_proposer = self._strategic_position_score(proposer_index)
        before_target = self._strategic_position_score(target_index)

        owners = [
            space.owner if space.is_ownable() else -2
            for space in self.board
        ]

        for prop_index in offer_properties:
            if 0 <= prop_index < len(owners):
                owners[prop_index] = target_index

        for prop_index in request_properties:
            if 0 <= prop_index < len(owners):
                owners[prop_index] = proposer_index

        after_proposer = self._estimated_strategic_position_score(
            proposer_index,
            owners,
            money_delta=request_money - offer_money,
        )
        after_target = self._estimated_strategic_position_score(
            target_index,
            owners,
            money_delta=offer_money - request_money,
        )

        return after_proposer - before_proposer, after_target - before_target

    def _estimated_strategic_position_score(
        self,
        player_index: int,
        owners: List[Optional[int]],
        money_delta: int = 0,
    ) -> float:
        net_worth = self._estimated_net_worth(player_index, owners, money_delta)
        completed = 0
        near = 0
        blockers = 0
        highest_rent = 0

        for group, properties in GROUPS.items():
            owned = sum(1 for prop_index in properties if owners[prop_index] == player_index)
            if owned == len(properties):
                completed += 1
            if owned == len(properties) - 1:
                near += 1

            for prop_index in properties:
                if owners[prop_index] != player_index:
                    continue
                for opponent_index, opponent in enumerate(self.players):
                    if opponent_index == player_index or opponent.bankrupt:
                        continue
                    opponent_owned = sum(1 for other_index in properties if owners[other_index] == opponent_index)
                    if opponent_owned == len(properties) - 1:
                        blockers += 1
                        break

        for prop_index, space in enumerate(self.board):
            if owners[prop_index] != player_index:
                continue
            if space.type == "property" and space.rent_schedule:
                highest_rent = max(highest_rent, max(space.rent_schedule))
            elif space.type == "railroad":
                highest_rent = max(highest_rent, 200)
            elif space.type == "utility":
                highest_rent = max(highest_rent, 120)

        return (
            net_worth
            + completed * 650.0
            + near * 180.0
            + blockers * 90.0
            + highest_rent * 0.45
            + self._total_houses(player_index) * 80.0
        )

    def _estimated_net_worth(
        self,
        player_index: int,
        owners: List[Optional[int]],
        money_delta: int = 0,
    ) -> int:
        total = self.players[player_index].money + money_delta
        for prop_index, space in enumerate(self.board):
            if owners[prop_index] == player_index:
                total += space.mortgage_value if space.mortgaged else space.price
                total += space.houses * space.build_cost
        return total

    def _property_completes_group_for_player(self, prop_index: int, player_index: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False
        space = self.board[prop_index]
        if space.group not in GROUPS:
            return False
        return all(
            self.board[group_prop].owner == player_index or group_prop == prop_index
            for group_prop in GROUPS[space.group]
        )

    def _property_creates_near_group_for_player(self, prop_index: int, player_index: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False
        space = self.board[prop_index]
        if space.group not in GROUPS:
            return False
        owned_after = sum(
            1
            for group_prop in GROUPS[space.group]
            if self.board[group_prop].owner == player_index or group_prop == prop_index
        )
        return owned_after == len(GROUPS[space.group]) - 1

    def _property_is_critical_for_owner(self, prop_index: int, owner_index: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False
        space = self.board[prop_index]
        if space.group not in GROUPS:
            return False
        owned = sum(1 for group_prop in GROUPS[space.group] if self.board[group_prop].owner == owner_index)
        return owned >= len(GROUPS[space.group]) - 1

    def _group_has_buildings(self, group: Optional[str]) -> bool:
        return bool(group and any(self.board[i].houses > 0 for i in GROUPS.get(group, [])))

    def _can_build_on_property(self, player_index: int, prop_index: int) -> bool:
        space = self.board[prop_index]
        player = self.players[player_index]

        if (
            space.type != "property"
            or space.owner != player_index
            or space.mortgaged
            or space.houses >= MAX_HOUSES
            or prop_index in self.properties_sold_buildings_this_turn
            or not self._owns_complete_group(player_index, space.group)
            or player.money < space.build_cost
        ):
            return False

        group_spaces = [self.board[i] for i in GROUPS[space.group]]
        if any(group_space.mortgaged for group_space in group_spaces):
            return False

        group_house_counts = [group_space.houses for group_space in group_spaces]
        if space.houses != min(group_house_counts):
            return False

        if space.houses == 4:
            return self.bank_hotels > 0

        return self.bank_houses > 0

    def _can_sell_building(self, player_index: int, prop_index: int) -> bool:
        space = self.board[prop_index]
        if (
            space.type != "property"
            or space.owner != player_index
            or space.houses <= 0
        ):
            return False

        if prop_index in self.properties_built_this_turn and self.players[player_index].money >= 0:
            return False

        group_house_counts = [self.board[i].houses for i in GROUPS[space.group]]
        if space.houses != max(group_house_counts):
            return False

        if space.houses == MAX_HOUSES and self.bank_houses < 4:
            return False

        return True

    def _can_mortgage_property(self, player_index: int, prop_index: int) -> bool:
        space = self.board[prop_index]
        if not space.is_ownable() or space.owner != player_index or space.mortgaged:
            return False
        if space.type == "property" and self._group_has_buildings(space.group):
            return False
        return True

    def _get_build_actions(self, player_index: int) -> List[Dict[str, Any]]:
        return [
            {"type": BUILD_HOUSE, "property_index": prop_index}
            for prop_index, space in enumerate(self.board)
            if space.type == "property" and self._can_build_on_property(player_index, prop_index)
        ]

    def _get_sell_house_actions(self, player_index: int) -> List[Dict[str, Any]]:
        return [
            {"type": SELL_HOUSE, "property_index": prop_index}
            for prop_index, space in enumerate(self.board)
            if space.type == "property" and self._can_sell_building(player_index, prop_index)
        ]

    def _get_mortgage_actions(self, player_index: int) -> List[Dict[str, Any]]:
        return [
            {"type": MORTGAGE_PROPERTY, "property_index": prop_index}
            for prop_index, space in enumerate(self.board)
            if space.is_ownable() and self._can_mortgage_property(player_index, prop_index)
        ]

    def _get_unmortgage_actions(self, player_index: int) -> List[Dict[str, Any]]:
        actions = []
        player = self.players[player_index]
        for prop_index, space in enumerate(self.board):
            cost = self._unmortgage_cost(space)
            if (
                space.owner == player_index
                and space.mortgaged
                and player.money >= cost + UNMORTGAGE_CASH_RESERVE
            ):
                actions.append({"type": UNMORTGAGE_PROPERTY, "property_index": prop_index})
        return actions

    def _unmortgage_cost(self, space: Space) -> int:
        return int(math.ceil(space.mortgage_value * 1.10))

    def _charge_mortgage_transfer_interest(self, owner_index: int, prop_index: int):
        space = self.board[prop_index]
        if space.mortgaged:
            self.players[owner_index].money -= int(math.ceil(space.mortgage_value * 0.10))
            self._settle_debt(owner_index, None)

    def _repair_cost(self, player_index: int, house_cost: int, hotel_cost: int) -> int:
        houses = 0
        hotels = 0
        for space in self.board:
            if space.owner != player_index or space.type != "property":
                continue
            if space.houses == MAX_HOUSES:
                hotels += 1
            else:
                houses += space.houses
        return houses * house_cost + hotels * hotel_cost

    def get_owned_properties(self, player_index: int) -> List[int]:
        return [
            i
            for i, space in enumerate(self.board)
            if space.is_ownable() and space.owner == player_index
        ]

    def get_tradeable_properties(self, player_index: int) -> List[int]:
        return [
            i
            for i, space in enumerate(self.board)
            if space.is_ownable() and self._is_tradeable_property(i, player_index)
        ]

    def describe_pending_trade(self) -> List[str]:
        if not self.pending_trade:
            return []

        return self._describe_trade(self.pending_trade, "Troca pendente:")

    def describe_trade_draft(self) -> List[str]:
        if not self.trade_draft:
            return []

        return self._describe_trade(
            self.trade_draft,
            f"Montando troca: {self.trade_draft.get('stage', '-')}",
        )

    def _describe_trade(self, trade: Dict[str, Any], title: str) -> List[str]:
        proposer = self.players[trade["from"]]
        target_index = trade.get("to")
        target_name = "alvo nao definido"
        if target_index is not None and 0 <= target_index < len(self.players):
            target_name = self.players[target_index].name

        lines = [title, f"{proposer.name} -> {target_name}", "", "Oferece:"]

        if trade["offer_properties"]:
            lines.extend(f"- {self.board[prop_index].name}" for prop_index in trade["offer_properties"])
        else:
            lines.append("- Nenhuma propriedade")

        lines.extend([f"- ${trade['offer_money']}", "", "Pede:"])

        if trade["request_properties"]:
            lines.extend(f"- {self.board[prop_index].name}" for prop_index in trade["request_properties"])
        else:
            lines.append("- Nenhuma propriedade")

        lines.append(f"- ${trade['request_money']}")
        return lines

    def _trade_response_action(self, action_type: str) -> Dict[str, Any]:
        assert self.pending_trade is not None
        trade = self.pending_trade
        action = {
            "type": action_type,
            "target_player": trade["from"],
            "offer_properties": list(trade["offer_properties"]),
            "offer_money": int(trade["offer_money"]),
            "request_properties": list(trade["request_properties"]),
            "request_money": int(trade["request_money"]),
        }
        return self._with_trade_features(action, trade["from"])

    def _with_trade_features(self, action: Dict[str, Any], proposer_index: int) -> Dict[str, Any]:
        if action.get("type") not in (PROPOSE_TRADE, ACCEPT_TRADE, DECLINE_TRADE):
            return action

        target_index = int(action.get("target_player", -1))
        if action.get("type") in (ACCEPT_TRADE, DECLINE_TRADE):
            target_index = self.pending_trade["to"] if self.pending_trade else target_index

        if not (0 <= proposer_index < self.num_players and 0 <= target_index < self.num_players):
            return action

        enriched = copy.deepcopy(action)
        enriched["trade_features"] = self._trade_feature_payload(
            proposer_index,
            target_index,
            list(enriched.get("offer_properties", [])),
            int(enriched.get("offer_money", 0)),
            list(enriched.get("request_properties", [])),
            int(enriched.get("request_money", 0)),
        )
        return enriched

    def _trade_feature_payload(
        self,
        proposer_index: int,
        target_index: int,
        offer_properties: List[int],
        offer_money: int,
        request_properties: List[int],
        request_money: int,
    ) -> Dict[str, float]:
        proposer_delta, target_delta = self._estimate_trade_score_deltas(
            proposer_index,
            target_index,
            offer_properties,
            offer_money,
            request_properties,
            request_money,
        )
        offer_value = sum(self._trade_property_value(target_index, prop_index) for prop_index in offer_properties)
        request_value = sum(self._trade_property_value(proposer_index, prop_index) for prop_index in request_properties)

        offer_completes_target = any(
            self._property_completes_group_for_player(prop_index, target_index)
            for prop_index in offer_properties
        )
        request_completes_proposer = any(
            self._property_completes_group_for_player(prop_index, proposer_index)
            for prop_index in request_properties
        )
        offer_near_target = any(
            self._property_creates_near_group_for_player(prop_index, target_index)
            for prop_index in offer_properties
        )
        request_near_proposer = any(
            self._property_creates_near_group_for_player(prop_index, proposer_index)
            for prop_index in request_properties
        )
        offer_critical_proposer = any(
            self._property_is_critical_for_owner(prop_index, proposer_index)
            for prop_index in offer_properties
        )

        return {
            "proposer_delta": proposer_delta / 1000.0,
            "target_delta": target_delta / 1000.0,
            "delta_diff": (proposer_delta - target_delta) / 1000.0,
            "offer_value": offer_value / 1000.0,
            "request_value": request_value / 1000.0,
            "offer_money": offer_money / 1500.0,
            "request_money": request_money / 1500.0,
            "offer_completes_target": 1.0 if offer_completes_target else 0.0,
            "request_completes_proposer": 1.0 if request_completes_proposer else 0.0,
            "offer_near_target": 1.0 if offer_near_target else 0.0,
            "request_near_proposer": 1.0 if request_near_proposer else 0.0,
            "offer_critical_proposer": 1.0 if offer_critical_proposer else 0.0,
        }

    def _trade_property_value(self, player_index: int, prop_index: int) -> int:
        if not (0 <= prop_index < len(self.board)):
            return 0

        space = self.board[prop_index]
        if not space.is_ownable():
            return 0

        value = space.price
        if space.type == "property" and space.group in GROUPS:
            if self._property_completes_group_for_player(prop_index, player_index):
                value = int(value * 2.0)
            elif self._property_creates_near_group_for_player(prop_index, player_index):
                value = int(value * 1.35)
            elif self._property_blocks_opponent_completion(prop_index, player_index):
                value = int(value * 1.65)
        elif space.type == "railroad":
            owned = sum(1 for i in RAILROADS if self.board[i].owner == player_index)
            value = int(value * (1.0 + 0.25 * owned))
        elif space.type == "utility":
            owned = sum(1 for i in UTILITIES if self.board[i].owner == player_index)
            value = int(value * (1.0 + 0.35 * owned))

        return value

    def _get_acting_player_for_action(self, action: Dict[str, Any]) -> int:
        if self.phase == "pending_trade_response" and self.pending_trade is not None:
            return self.pending_trade["to"]
        if self.phase == "auction" and self.auction is not None:
            return self.auction["current_bidder"]
        return self.current_player_index

    def _register_event(self, acting_player_index: int, action: Dict[str, Any], reward: float):
        self.action_count += 1
        player_name = self.players[acting_player_index].name
        event = {
            "number": self.action_count,
            "turn": self.turn_count,
            "player": player_name,
            "action_type": action.get("type", "unknown"),
            "message": self.last_message,
            "reward": reward,
        }
        self.event_history.append(event)
        if len(self.event_history) > 12:
            self.event_history = self.event_history[-12:]
