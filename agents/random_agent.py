import random
from typing import Any, Dict, List, Optional

from game.actions import (
    ACCEPT_TRADE,
    ADD_TRADE_OFFER_PROPERTY,
    ADD_TRADE_REQUEST_PROPERTY,
    AUCTION_BID,
    BUILD_HOUSE,
    BUY_PROPERTY,
    CANCEL_TRADE,
    DECLINE_TRADE,
    FINISH_TRADE_OFFER,
    FINISH_TRADE_REQUEST,
    MORTGAGE_PROPERTY,
    PAY_JAIL_FINE,
    ROLL_DICE,
    SELECT_TRADE_TARGET,
    SET_TRADE_OFFER_MONEY,
    SET_TRADE_REQUEST_MONEY,
    START_TRADE,
    SUBMIT_TRADE,
    SELL_HOUSE,
    UNMORTGAGE_PROPERTY,
    USE_JAIL_CARD,
)
from game.board import GROUPS, RAILROADS, UTILITIES, create_board

class RandomAgent:
    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)
        self.board = create_board()

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        trade_action = self._choose_trade_action(state, valid_actions)
        if trade_action is not None:
            return trade_action

        buy_actions = [a for a in valid_actions if a["type"] == BUY_PROPERTY]
        if buy_actions and self.random.random() < 0.95:
            return buy_actions[0]

        jail_card_actions = [a for a in valid_actions if a["type"] == USE_JAIL_CARD]
        if jail_card_actions and self.random.random() < 0.75:
            return jail_card_actions[0]

        jail_pay_actions = [a for a in valid_actions if a["type"] == PAY_JAIL_FINE]
        if jail_pay_actions and self.random.random() < 0.35:
            return jail_pay_actions[0]

        build_actions = [a for a in valid_actions if a["type"] == BUILD_HOUSE]
        if build_actions and self.random.random() < 0.70:
            return self.random.choice(build_actions)

        unmortgage_actions = [a for a in valid_actions if a["type"] == UNMORTGAGE_PROPERTY]
        if unmortgage_actions and self.random.random() < 0.20:
            return self.random.choice(unmortgage_actions)

        auction_bid_actions = [a for a in valid_actions if a["type"] == AUCTION_BID]
        if auction_bid_actions and self.random.random() < 0.65:
            return self.random.choice(auction_bid_actions)

        roll_actions = [a for a in valid_actions if a["type"] == ROLL_DICE]
        if roll_actions:
            return roll_actions[0]

        # Mortgage/sell actions are mostly defensive; leave them for low-cash
        # situations or random exploration.
        money = self._money(state, self.player_id)
        defensive_actions = [
            a
            for a in valid_actions
            if a["type"] in (MORTGAGE_PROPERTY, SELL_HOUSE)
        ]
        if defensive_actions and money < 150:
            return self.random.choice(defensive_actions)

        cancel_trade_actions = [a for a in valid_actions if a["type"] == CANCEL_TRADE]
        if cancel_trade_actions and len(valid_actions) == 1:
            return cancel_trade_actions[0]

        return self.random.choice(valid_actions)

    # ========================================================
    # Trade heuristics
    # ========================================================

    def _choose_trade_action(
        self,
        state: Dict[str, Any],
        valid_actions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        action_types = {action["type"] for action in valid_actions}
        trade_types = {
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
            ACCEPT_TRADE,
            DECLINE_TRADE,
        }
        if not action_types.intersection(trade_types):
            return None

        if action_types.intersection({ACCEPT_TRADE, DECLINE_TRADE}):
            return self._choose_trade_response(state, valid_actions)

        if START_TRADE in action_types:
            start_action = self._only_action(valid_actions, START_TRADE)
            if start_action and self._should_start_trade(state):
                return start_action
            return None

        if CANCEL_TRADE in action_types and len(valid_actions) == 1:
            return valid_actions[0]

        if SELECT_TRADE_TARGET in action_types:
            return self._choose_trade_target(state, valid_actions)

        if ADD_TRADE_OFFER_PROPERTY in action_types or FINISH_TRADE_OFFER in action_types:
            return self._choose_offer_property_step(state, valid_actions)

        if SET_TRADE_OFFER_MONEY in action_types:
            return self._choose_offer_money_step(state, valid_actions)

        if ADD_TRADE_REQUEST_PROPERTY in action_types or FINISH_TRADE_REQUEST in action_types:
            return self._choose_request_property_step(state, valid_actions)

        if SET_TRADE_REQUEST_MONEY in action_types:
            return self._choose_request_money_step(state, valid_actions)

        if SUBMIT_TRADE in action_types:
            return self._choose_submit_trade(state, valid_actions)

        return None

    def _should_start_trade(self, state: Dict[str, Any]) -> bool:
        player_id = self._acting_player(state)
        target_scores = [
            self._target_trade_score(state, player_id, target_id)
            for target_id in range(4)
            if target_id != player_id and not self._bankrupt(state, target_id)
        ]
        if not target_scores:
            return False
        best_target_score = max(target_scores)
        if best_target_score >= 3.0:
            return self.random.random() < 0.95
        if best_target_score >= 2.0:
            return self.random.random() < 0.75
        if best_target_score >= 1.0:
            return self.random.random() < 0.30
        return False

    def _choose_trade_target(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        player_id = self._acting_player(state)
        target_actions = [a for a in valid_actions if a["type"] == SELECT_TRADE_TARGET]
        return max(
            target_actions,
            key=lambda action: self._target_trade_score(state, player_id, int(action.get("target_player", -1))),
        )

    def _choose_offer_property_step(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        finish_action = self._only_action(valid_actions, FINISH_TRADE_OFFER)
        add_actions = [a for a in valid_actions if a["type"] == ADD_TRADE_OFFER_PROPERTY]
        if not add_actions:
            return finish_action or valid_actions[0]

        draft = state.get("trade_draft") or {}
        player_id = int(draft.get("from", self._acting_player(state)))
        target_id = self._int_or_default(draft.get("to"), -1)
        selected = set(draft.get("offer_properties", []))
        if selected:
            return finish_action or valid_actions[0]

        best_action = max(
            add_actions,
            key=lambda action: self._offer_property_score(
                state,
                player_id,
                target_id,
                int(action.get("property_index", -1)),
            ),
        )
        best_score = self._offer_property_score(
            state,
            player_id,
            target_id,
            int(best_action.get("property_index", -1)),
        )
        if best_score > 0.0:
            return best_action
        return finish_action or best_action

    def _choose_offer_money_step(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        draft = state.get("trade_draft") or {}
        player_id = int(draft.get("from", self._acting_player(state)))
        target_id = self._int_or_default(draft.get("to"), -1)

        desired_property = self._best_property_to_request(state, player_id, target_id)
        offer_properties = list(draft.get("offer_properties", []))
        current_offer_value = sum(
            self._property_value_for_player(state, target_id, prop_index)
            for prop_index in offer_properties
        )

        if desired_property is None:
            desired_amount = 0
        else:
            desired_value_for_self = self._property_value_for_player(state, player_id, desired_property)
            desired_value_for_target = self._property_value_for_player(state, target_id, desired_property)
            target_required_value = desired_value_for_target
            if self._property_completes_group_for_player(state, desired_property, player_id):
                target_required_value += 120.0

            desired_value = max(
                desired_value_for_self * 0.75,
                target_required_value,
            )
            desired_amount = max(0, int(desired_value - current_offer_value))
            desired_amount = min(desired_amount, max(0, self._money(state, player_id) - 250))

        return self._closest_amount_action(valid_actions, SET_TRADE_OFFER_MONEY, desired_amount)

    def _choose_request_property_step(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        finish_action = self._only_action(valid_actions, FINISH_TRADE_REQUEST)
        add_actions = [a for a in valid_actions if a["type"] == ADD_TRADE_REQUEST_PROPERTY]
        if not add_actions:
            return finish_action or valid_actions[0]

        draft = state.get("trade_draft") or {}
        if draft.get("request_properties"):
            return finish_action or valid_actions[0]

        player_id = int(draft.get("from", self._acting_player(state)))
        target_id = self._int_or_default(draft.get("to"), -1)
        best_action = max(
            add_actions,
            key=lambda action: self._request_property_score(
                state,
                player_id,
                target_id,
                int(action.get("property_index", -1)),
            ),
        )
        best_score = self._request_property_score(
            state,
            player_id,
            target_id,
            int(best_action.get("property_index", -1)),
        )
        if best_score > 0.0:
            return best_action
        return finish_action or best_action

    def _choose_request_money_step(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        draft = state.get("trade_draft") or {}
        player_id = int(draft.get("from", self._acting_player(state)))
        target_id = self._int_or_default(draft.get("to"), -1)

        offer_value = int(draft.get("offer_money", 0))
        offer_value += sum(
            self._property_value_for_player(state, target_id, prop_index)
            for prop_index in draft.get("offer_properties", [])
        )
        request_value = sum(
            self._property_value_for_player(state, player_id, prop_index)
            for prop_index in draft.get("request_properties", [])
        )

        desired_amount = max(0, int(offer_value - request_value))
        desired_amount = min(desired_amount, max(0, self._money(state, target_id) - 200))
        return self._closest_amount_action(valid_actions, SET_TRADE_REQUEST_MONEY, desired_amount)

    def _choose_submit_trade(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        submit_actions = [a for a in valid_actions if a["type"] == SUBMIT_TRADE]
        if not submit_actions:
            return valid_actions[0]
        return max(submit_actions, key=lambda action: self._proposer_trade_delta(state, action))

    def _choose_trade_response(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        accept_action = self._only_action(valid_actions, ACCEPT_TRADE)
        decline_action = self._only_action(valid_actions, DECLINE_TRADE)
        if accept_action is None:
            return decline_action or valid_actions[0]
        if decline_action is None:
            return accept_action

        responder_delta = self._responder_trade_delta(state, accept_action)
        proposer_delta = self._proposer_trade_delta(state, accept_action)
        if self._is_mutual_completion_trade(state, accept_action):
            if responder_delta >= -80.0 and proposer_delta <= responder_delta + 250.0:
                return accept_action

        if responder_delta >= 40.0 and responder_delta + 100.0 >= proposer_delta:
            return accept_action
        if responder_delta >= 120.0:
            return accept_action
        return decline_action

    def _is_mutual_completion_trade(self, state: Dict[str, Any], action: Dict[str, Any]) -> bool:
        responder_id = self._acting_player(state)
        proposer_id = self._trade_proposer(state, action)
        offer_props = list(action.get("offer_properties", []))
        request_props = list(action.get("request_properties", []))
        responder_gets_group = any(
            self._property_completes_group_for_player(state, prop, responder_id)
            for prop in offer_props
        )
        proposer_gets_group = any(
            self._property_completes_group_for_player(state, prop, proposer_id)
            for prop in request_props
        )
        return responder_gets_group and proposer_gets_group

    # ========================================================
    # Valuation helpers
    # ========================================================

    def _target_trade_score(self, state: Dict[str, Any], player_id: int, target_id: int) -> float:
        if target_id < 0 or target_id == player_id or self._bankrupt(state, target_id):
            return -999.0

        score = 0.0
        for prop_index, owner in enumerate(self._owners(state)):
            if owner == target_id and self._property_completes_group_for_player(state, prop_index, player_id):
                score += 2.0
            if owner == target_id:
                score += self._collection_upgrade_score(state, player_id, prop_index)
            if owner == player_id and self._property_completes_group_for_player(state, prop_index, target_id):
                score += 1.6
            if owner == player_id:
                score += 0.8 * self._collection_upgrade_score(state, target_id, prop_index)

        return score

    def _offer_property_score(self, state: Dict[str, Any], player_id: int, target_id: int, prop_index: int) -> float:
        if prop_index < 0:
            return -999.0
        value_to_target = self._property_value_for_player(state, target_id, prop_index)
        value_to_self = self._property_value_for_player(state, player_id, prop_index)
        score = value_to_target - value_to_self
        if self._property_completes_group_for_player(state, prop_index, target_id):
            score += 180.0
        if self._property_completes_group_for_player(state, prop_index, player_id):
            score -= 500.0
        return score

    def _request_property_score(self, state: Dict[str, Any], player_id: int, target_id: int, prop_index: int) -> float:
        if prop_index < 0:
            return -999.0
        value_to_self = self._property_value_for_player(state, player_id, prop_index)
        value_to_target = self._property_value_for_player(state, target_id, prop_index)
        score = value_to_self - 0.65 * value_to_target
        if self._property_completes_group_for_player(state, prop_index, player_id):
            score += 350.0
        if self._property_completes_group_for_player(state, prop_index, target_id):
            score -= 220.0
        return score

    def _proposer_trade_delta(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        proposer_id = self._trade_proposer(state, action)
        target_id = self._trade_target(state, action)
        offer_props = list(action.get("offer_properties", []))
        request_props = list(action.get("request_properties", []))

        gain = int(action.get("request_money", 0))
        gain += sum(self._property_value_for_player(state, proposer_id, prop) for prop in request_props)

        loss = int(action.get("offer_money", 0))
        loss += sum(self._property_value_for_player(state, proposer_id, prop) for prop in offer_props)

        if target_id >= 0:
            for prop in offer_props:
                if self._property_completes_group_for_player(state, prop, target_id):
                    loss += 120.0

        return gain - loss

    def _responder_trade_delta(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        responder_id = self._acting_player(state)
        proposer_id = self._trade_proposer(state, action)
        offer_props = list(action.get("offer_properties", []))
        request_props = list(action.get("request_properties", []))

        gain = int(action.get("offer_money", 0))
        gain += sum(self._property_value_for_player(state, responder_id, prop) for prop in offer_props)

        loss = int(action.get("request_money", 0))
        loss += sum(self._property_value_for_player(state, responder_id, prop) for prop in request_props)

        if proposer_id >= 0:
            for prop in request_props:
                if self._property_completes_group_for_player(state, prop, proposer_id):
                    loss += 120.0

        return gain - loss

    def _property_value_for_player(self, state: Dict[str, Any], player_id: int, prop_index: int) -> float:
        if not (0 <= prop_index < len(self.board)):
            return 0.0

        space = self.board[prop_index]
        if not space.is_ownable():
            return 0.0

        value = float(space.price)
        if self._mortgaged(state, prop_index):
            value *= 0.60

        if space.group in GROUPS:
            owned = self._owned_in_group(state, player_id, space.group)
            group_size = len(GROUPS[space.group])
            if self._property_completes_group_for_player(state, prop_index, player_id):
                value *= 2.60
            elif owned == group_size - 2:
                value *= 1.50
            elif owned > 0:
                value *= 1.15

            if self._property_blocks_any_opponent(state, prop_index, player_id):
                value *= 1.35

        elif prop_index in RAILROADS:
            owned = sum(1 for i in RAILROADS if self._owner(state, i) == player_id)
            value *= 1.0 + 0.25 * owned
        elif prop_index in UTILITIES:
            owned = sum(1 for i in UTILITIES if self._owner(state, i) == player_id)
            value *= 1.0 + 0.35 * owned

        return value

    def _best_property_to_request(self, state: Dict[str, Any], player_id: int, target_id: int) -> Optional[int]:
        best_prop = None
        best_score = float("-inf")
        for prop_index, owner in enumerate(self._owners(state)):
            if owner != target_id:
                continue
            score = self._request_property_score(state, player_id, target_id, prop_index)
            if score > best_score:
                best_score = score
                best_prop = prop_index
        return best_prop if best_score > 0 else None

    def _property_completes_group_for_player(self, state: Dict[str, Any], prop_index: int, player_id: int) -> bool:
        if not (0 <= prop_index < len(self.board)):
            return False
        group = self.board[prop_index].group
        if group not in GROUPS:
            return False
        return all(
            self._owner(state, group_prop) == player_id or group_prop == prop_index
            for group_prop in GROUPS[group]
        )

    def _property_blocks_any_opponent(self, state: Dict[str, Any], prop_index: int, player_id: int) -> bool:
        for opponent_id in range(4):
            if opponent_id == player_id or self._bankrupt(state, opponent_id):
                continue
            if self._property_completes_group_for_player(state, prop_index, opponent_id):
                return True
        return False

    def _owned_in_group(self, state: Dict[str, Any], player_id: int, group: str) -> int:
        return sum(1 for prop_index in GROUPS[group] if self._owner(state, prop_index) == player_id)

    def _collection_upgrade_score(self, state: Dict[str, Any], player_id: int, prop_index: int) -> float:
        if prop_index in RAILROADS:
            owned = sum(1 for railroad_index in RAILROADS if self._owner(state, railroad_index) == player_id)
            if owned >= 3:
                return 1.20
            if owned == 2:
                return 0.80
            if owned == 1:
                return 0.35
        if prop_index in UTILITIES:
            owned = sum(1 for utility_index in UTILITIES if self._owner(state, utility_index) == player_id)
            if owned == 1:
                return 0.65
        return 0.0

    # ========================================================
    # State helpers
    # ========================================================

    def _only_action(self, valid_actions: List[Dict[str, Any]], action_type: str) -> Optional[Dict[str, Any]]:
        for action in valid_actions:
            if action["type"] == action_type:
                return action
        return None

    def _closest_amount_action(
        self,
        valid_actions: List[Dict[str, Any]],
        action_type: str,
        desired_amount: int,
    ) -> Dict[str, Any]:
        amount_actions = [a for a in valid_actions if a["type"] == action_type]
        if not amount_actions:
            return valid_actions[0]
        return min(
            amount_actions,
            key=lambda action: abs(int(action.get("amount", action.get("offer_money", action.get("request_money", 0)))) - desired_amount),
        )

    def _acting_player(self, state: Dict[str, Any]) -> int:
        pending_trade = state.get("pending_trade")
        if isinstance(pending_trade, dict):
            responder_id = self._int_or_default(pending_trade.get("to"), -1)
            if responder_id >= 0:
                return responder_id

        auction = state.get("auction")
        if isinstance(auction, dict):
            bidder_id = self._int_or_default(auction.get("current_bidder"), -1)
            if bidder_id >= 0:
                return bidder_id

        return self._int_or_default(state.get("current_player", self.player_id), self.player_id)

    def _trade_proposer(self, state: Dict[str, Any], action: Dict[str, Any]) -> int:
        pending_trade = state.get("pending_trade")
        if isinstance(pending_trade, dict):
            return self._int_or_default(pending_trade.get("from"), -1)
        draft = state.get("trade_draft")
        if isinstance(draft, dict):
            return self._int_or_default(draft.get("from"), -1)
        return self._int_or_default(action.get("target_player"), -1)

    def _trade_target(self, state: Dict[str, Any], action: Dict[str, Any]) -> int:
        pending_trade = state.get("pending_trade")
        if isinstance(pending_trade, dict):
            return self._int_or_default(pending_trade.get("to"), -1)
        draft = state.get("trade_draft")
        if isinstance(draft, dict):
            return self._int_or_default(draft.get("to"), -1)
        return self._int_or_default(action.get("target_player"), -1)

    def _owners(self, state: Dict[str, Any]) -> List[int]:
        return [self._int_or_default(owner, -2) for owner in state.get("property_owners", [])]

    def _owner(self, state: Dict[str, Any], prop_index: int) -> int:
        owners = self._owners(state)
        if 0 <= prop_index < len(owners):
            return owners[prop_index]
        return -2

    def _money(self, state: Dict[str, Any], player_id: int) -> int:
        money = state.get("player_money", [])
        if 0 <= player_id < len(money):
            return int(money[player_id])
        return 0

    def _bankrupt(self, state: Dict[str, Any], player_id: int) -> bool:
        bankrupt = state.get("player_bankrupt", [])
        if 0 <= player_id < len(bankrupt):
            return bool(bankrupt[player_id])
        return True

    def _mortgaged(self, state: Dict[str, Any], prop_index: int) -> bool:
        mortgaged = state.get("property_mortgaged", [])
        if 0 <= prop_index < len(mortgaged):
            return bool(mortgaged[prop_index])
        return False

    def _int_or_default(self, value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
