import random
from typing import Any, Dict, List

from game.actions import ACCEPT_TRADE, BUILD_HOUSE, BUY_PROPERTY, DECLINE_TRADE, PROPOSE_TRADE, ROLL_DICE


class RandomAgent:
    """
    Agente simples para testar o ambiente.

    Ele escolhe ações válidas de forma aleatória, mas com alguns pesos básicos
    para o jogo andar melhor visualmente:
    - geralmente compra propriedades quando pode;
    - constrói casas/hotel com frequência quando fecha uma região;
    - raramente propõe trocas;
    - aceita ou recusa trocas aleatoriamente.
    """

    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        buy_actions = [a for a in valid_actions if a["type"] == BUY_PROPERTY]
        if buy_actions and self.random.random() < 0.98:
            return buy_actions[0]

        build_actions = [a for a in valid_actions if a["type"] == BUILD_HOUSE]
        if build_actions and self.random.random() < 0.95:
            return self.random.choice(build_actions)

        trade_actions = [a for a in valid_actions if a["type"] == PROPOSE_TRADE]
        roll_actions = [a for a in valid_actions if a["type"] == ROLL_DICE]

        # Evita que os agentes fiquem propondo troca demais.
        if trade_actions and self.random.random() < 0.45:
            return self.random.choice(trade_actions)

        if roll_actions:
            return roll_actions[0]

        accept_actions = [a for a in valid_actions if a["type"] == ACCEPT_TRADE]
        if accept_actions and self.random.random() < 0.75:
            return accept_actions[0]

        decline_actions = [a for a in valid_actions if a["type"] == DECLINE_TRADE]
        if decline_actions and self.random.random() < 0.15:
            return decline_actions[0]

        return self.random.choice(valid_actions)
