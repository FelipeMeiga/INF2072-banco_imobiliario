import random
from typing import Any, Dict, List

from game.actions import BUY_PROPERTY, DECLINE_TRADE, PROPOSE_TRADE, ROLL_DICE


class RandomAgent:

    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        buy_actions = [a for a in valid_actions if a["type"] == BUY_PROPERTY]
        if buy_actions and self.random.random() < 0.75:
            return buy_actions[0]

        trade_actions = [a for a in valid_actions if a["type"] == PROPOSE_TRADE]
        roll_actions = [a for a in valid_actions if a["type"] == ROLL_DICE]

        if trade_actions and self.random.random() < 0.10:
            return self.random.choice(trade_actions)

        if roll_actions:
            return roll_actions[0]

        decline_actions = [a for a in valid_actions if a["type"] == DECLINE_TRADE]
        if decline_actions and self.random.random() < 0.50:
            return decline_actions[0]

        return self.random.choice(valid_actions)
