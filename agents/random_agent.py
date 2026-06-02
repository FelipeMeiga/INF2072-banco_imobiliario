import random
from typing import Any, Dict, List

from game.actions import (
    ACCEPT_TRADE,
    AUCTION_BID,
    BUILD_HOUSE,
    BUY_PROPERTY,
    DECLINE_TRADE,
    MORTGAGE_PROPERTY,
    PAY_JAIL_FINE,
    PROPOSE_TRADE,
    ROLL_DICE,
    SELL_HOUSE,
    UNMORTGAGE_PROPERTY,
    USE_JAIL_CARD,
)


class RandomAgent:
    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

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

        trade_actions = [a for a in valid_actions if a["type"] == PROPOSE_TRADE]
        if trade_actions and self.random.random() < 0.25:
            return self.random.choice(trade_actions)

        accept_actions = [a for a in valid_actions if a["type"] == ACCEPT_TRADE]
        if accept_actions and self.random.random() < 0.65:
            return accept_actions[0]

        roll_actions = [a for a in valid_actions if a["type"] == ROLL_DICE]
        if roll_actions:
            return roll_actions[0]

        decline_actions = [a for a in valid_actions if a["type"] == DECLINE_TRADE]
        if decline_actions and self.random.random() < 0.20:
            return decline_actions[0]

        # Mortgage/sell actions are mostly defensive; leave them for low-cash
        # situations or random exploration.
        money = state.get("player_money", [0, 0, 0, 0])[self.player_id]
        defensive_actions = [
            a
            for a in valid_actions
            if a["type"] in (MORTGAGE_PROPERTY, SELL_HOUSE)
        ]
        if defensive_actions and money < 150:
            return self.random.choice(defensive_actions)

        return self.random.choice(valid_actions)
