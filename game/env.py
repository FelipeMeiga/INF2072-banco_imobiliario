import random
from typing import Any, Dict, List, Optional

from game.actions import (
    ACCEPT_TRADE,
    BUY_PROPERTY,
    DECLINE_TRADE,
    PASS_BUY,
    PROPOSE_TRADE,
    ROLL_DICE,
)
from game.board import NUM_SPACES, create_board
from game.models import Player, Space

START_MONEY = 1500
PASS_START_BONUS = 200
MAX_TURNS = 500

PLAYER_COLORS = [
    (70, 120, 220),
    (210, 60, 60),
    (50, 170, 80),
    (150, 80, 200),
]


class BancoImobiliarioEnv:

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        self.num_players = num_players
        self.seed = seed
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

        self.current_player_index = 0
        self.dice = (1, 1)
        self.phase = "ready_to_roll"  # ready_to_roll, awaiting_buy, pending_trade_response, game_over
        self.pending_trade: Optional[Dict[str, Any]] = None
        self.trade_proposed_this_turn = False
        self.done = False
        self.winner: Optional[int] = None
        self.turn_count = 0
        self.last_message = "Jogo iniciado."
        self.last_action: Optional[Dict[str, Any]] = None
        return self.get_state()

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_player": self.current_player_index,
            "phase": self.phase,
            "dice": self.dice,
            "player_money": [p.money for p in self.players],
            "player_positions": [p.position for p in self.players],
            "player_bankrupt": [p.bankrupt for p in self.players],
            "property_owners": [
                space.owner if space.type == "property" else -2
                for space in self.board
            ],
            "pending_trade": self.pending_trade,
            "done": self.done,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "last_message": self.last_message,
        }

    def get_valid_actions(self, player_index: Optional[int] = None) -> List[Dict[str, Any]]:
        if player_index is None:
            player_index = self.current_player_index

        if self.done:
            return []

        if self.phase == "pending_trade_response":
            if self.pending_trade and self.pending_trade["to"] == player_index:
                return [
                    {"type": ACCEPT_TRADE},
                    {"type": DECLINE_TRADE},
                ]
            return []

        if player_index != self.current_player_index:
            return []

        if self.phase == "awaiting_buy":
            return [
                {"type": BUY_PROPERTY},
                {"type": PASS_BUY},
            ]

        if self.phase == "ready_to_roll":
            actions = [{"type": ROLL_DICE}]

            # Para não explodir o espaço de ações, geramos algumas propostas candidatas.
            # O agente aleatório pode escolher uma delas ou simplesmente rolar os dados.
            if not self.trade_proposed_this_turn:
                actions.extend(self._sample_trade_actions(player_index, max_actions=6))

            return actions

        return []

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.get_state(), 0.0, self.done, {"message": self.last_message}

        self.last_action = action
        before_net_worth = self._net_worth(self.current_player_index)
        action_type = action.get("type")

        if action_type == ROLL_DICE:
            self._roll_dice()
        elif action_type == BUY_PROPERTY:
            self._buy_property()
        elif action_type == PASS_BUY:
            self._pass_buy()
        elif action_type == PROPOSE_TRADE:
            self._propose_trade(action)
        elif action_type == ACCEPT_TRADE:
            self._accept_trade()
        elif action_type == DECLINE_TRADE:
            self._decline_trade()
        else:
            self.last_message = f"Ação inválida: {action_type}"

        self._check_game_end()

        after_net_worth = self._net_worth(self.current_player_index)
        reward = (after_net_worth - before_net_worth) / 100.0

        if self.done and self.winner is not None:
            reward += 100.0 if self.winner == self.current_player_index else -100.0

        return self.get_state(), reward, self.done, {"message": self.last_message}

    # ========================================================
    # AÇÕES PRINCIPAIS
    # ========================================================

    def _roll_dice(self):
        if self.phase != "ready_to_roll":
            self.last_message = "Não é possível rolar os dados agora."
            return

        player = self.current_player

        if player.bankrupt:
            self._next_turn()
            return

        d1 = self.random.randint(1, 6)
        d2 = self.random.randint(1, 6)
        total = d1 + d2
        self.dice = (d1, d2)

        passed_start = player.move(total, NUM_SPACES)

        if passed_start:
            player.money += PASS_START_BONUS

        self._resolve_landing()

    def _resolve_landing(self):
        player = self.current_player
        space = self.board[player.position]

        if space.type == "start":
            self.last_message = f"{player.name} caiu no início."
            self._next_turn()
            return

        if space.type == "property":
            if space.owner is None:
                self.phase = "awaiting_buy"
                self.last_message = (
                    f"{player.name} caiu em {space.name}. "
                    f"Preço: ${space.price}. Aluguel: ${space.rent}."
                )
                return

            if space.owner == self.current_player_index:
                self.last_message = f"{player.name} caiu na própria propriedade: {space.name}."
                self._next_turn()
                return

            owner = self.players[space.owner]
            player.money -= space.rent
            owner.money += space.rent
            self.last_message = (
                f"{player.name} pagou ${space.rent} de aluguel para "
                f"{owner.name} por {space.name}."
            )
            self._check_bankruptcy(player)
            self._next_turn()
            return

        if space.type == "tax":
            tax = space.rent
            player.money -= tax
            self.last_message = f"{player.name} pagou imposto de ${tax}."
            self._check_bankruptcy(player)
            self._next_turn()
            return

        if space.type == "chance":
            amount = self.random.choice([-100, -50, 50, 100, 150])
            player.money += amount

            if amount >= 0:
                self.last_message = f"Sorte! {player.name} recebeu ${amount}."
            else:
                self.last_message = f"Azar! {player.name} perdeu ${abs(amount)}."

            self._check_bankruptcy(player)
            self._next_turn()
            return

        if space.type == "jail":
            self.last_message = f"{player.name} visitou a prisão. Por enquanto, nada acontece."
            self._next_turn()
            return

        self.last_message = f"{player.name} caiu em uma casa livre."
        self._next_turn()

    def _buy_property(self):
        if self.phase != "awaiting_buy":
            self.last_message = "Não existe propriedade para comprar agora."
            return

        player = self.current_player
        space = self.board[player.position]

        if space.type != "property" or space.owner is not None:
            self.phase = "ready_to_roll"
            self._next_turn()
            return

        if player.money >= space.price:
            player.money -= space.price
            space.owner = self.current_player_index
            self.last_message = f"{player.name} comprou {space.name} por ${space.price}."
        else:
            self.last_message = f"{player.name} não tinha dinheiro para comprar {space.name}."

        self.phase = "ready_to_roll"
        self._next_turn()

    def _pass_buy(self):
        if self.phase != "awaiting_buy":
            self.last_message = "Não existe compra para passar agora."
            return

        player = self.current_player
        space = self.board[player.position]
        self.last_message = f"{player.name} decidiu não comprar {space.name}."
        self.phase = "ready_to_roll"
        self._next_turn()

    # ========================================================
    # TROCAS
    # ========================================================

    def _propose_trade(self, action: Dict[str, Any]):
        if self.phase != "ready_to_roll":
            self.last_message = "Trocas só podem ser propostas antes de rolar os dados."
            return

        if self.trade_proposed_this_turn:
            self.last_message = "Esse jogador já propôs uma troca neste turno."
            return

        proposer_index = self.current_player_index
        target_index = action.get("target_player")

        if target_index is None or target_index == proposer_index:
            self.last_message = "Jogador alvo inválido para troca."
            return

        if target_index < 0 or target_index >= len(self.players):
            self.last_message = "Jogador alvo inexistente."
            return

        proposer = self.players[proposer_index]
        target = self.players[target_index]

        if proposer.bankrupt or target.bankrupt:
            self.last_message = "Jogador falido não pode participar de troca."
            return

        offer_properties = list(action.get("offer_properties", []))
        request_properties = list(action.get("request_properties", []))
        offer_money = int(action.get("offer_money", 0))
        request_money = int(action.get("request_money", 0))

        if offer_money < 0 or request_money < 0:
            self.last_message = "Valores de dinheiro na troca não podem ser negativos."
            return

        if (
            not offer_properties
            and not request_properties
            and offer_money == 0
            and request_money == 0
        ):
            self.last_message = "Troca vazia ignorada."
            return

        if offer_money > proposer.money:
            self.last_message = f"{proposer.name} não tem dinheiro suficiente para essa oferta."
            return

        if request_money > target.money:
            self.last_message = f"{target.name} não tem dinheiro suficiente para pagar o valor pedido."
            return

        for prop_index in offer_properties:
            if not self._is_valid_property_owner(prop_index, proposer_index):
                self.last_message = "Uma propriedade oferecida não pertence ao jogador atual."
                return

        for prop_index in request_properties:
            if not self._is_valid_property_owner(prop_index, target_index):
                self.last_message = "Uma propriedade pedida não pertence ao jogador alvo."
                return

        self.pending_trade = {
            "from": proposer_index,
            "to": target_index,
            "offer_properties": offer_properties,
            "request_properties": request_properties,
            "offer_money": offer_money,
            "request_money": request_money,
        }

        self.phase = "pending_trade_response"
        self.trade_proposed_this_turn = True
        self.last_message = f"{proposer.name} propôs uma troca para {target.name}."

    def _accept_trade(self):
        if self.phase != "pending_trade_response" or self.pending_trade is None:
            self.last_message = "Não existe troca pendente."
            return

        trade = self.pending_trade
        proposer_index = trade["from"]
        target_index = trade["to"]
        proposer = self.players[proposer_index]
        target = self.players[target_index]

        if proposer.bankrupt or target.bankrupt:
            self.last_message = "Troca cancelada porque um jogador faliu."
            self._clear_trade()
            return

        if trade["offer_money"] > proposer.money or trade["request_money"] > target.money:
            self.last_message = "Troca cancelada por dinheiro insuficiente."
            self._clear_trade()
            return

        for prop_index in trade["offer_properties"]:
            if not self._is_valid_property_owner(prop_index, proposer_index):
                self.last_message = "Troca cancelada: propriedade oferecida mudou de dono."
                self._clear_trade()
                return

        for prop_index in trade["request_properties"]:
            if not self._is_valid_property_owner(prop_index, target_index):
                self.last_message = "Troca cancelada: propriedade pedida mudou de dono."
                self._clear_trade()
                return

        proposer.money -= trade["offer_money"]
        target.money += trade["offer_money"]

        target.money -= trade["request_money"]
        proposer.money += trade["request_money"]

        for prop_index in trade["offer_properties"]:
            self.board[prop_index].owner = target_index

        for prop_index in trade["request_properties"]:
            self.board[prop_index].owner = proposer_index

        self.last_message = f"{target.name} aceitou a troca de {proposer.name}."
        self._clear_trade()

    def _decline_trade(self):
        if self.phase != "pending_trade_response" or self.pending_trade is None:
            self.last_message = "Não existe troca pendente."
            return

        proposer = self.players[self.pending_trade["from"]]
        target = self.players[self.pending_trade["to"]]
        self.last_message = f"{target.name} recusou a troca de {proposer.name}."
        self._clear_trade()

    def _clear_trade(self):
        self.pending_trade = None
        self.phase = "ready_to_roll"

    # ========================================================
    # HELPERS
    # ========================================================

    def _next_turn(self):
        if self.done:
            return

        self.phase = "ready_to_roll"
        self.trade_proposed_this_turn = False
        self.turn_count += 1

        if self.turn_count >= MAX_TURNS:
            self._finish_by_net_worth()
            return

        for _ in range(len(self.players)):
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            if not self.current_player.bankrupt:
                return

    def _check_bankruptcy(self, player: Player):
        if player.money >= 0 or player.bankrupt:
            return

        player.bankrupt = True
        player_index = self.players.index(player)

        for space in self.board:
            if space.owner == player_index:
                space.owner = None

        self.last_message += f" {player.name} faliu!"

    def _check_game_end(self):
        active_players = [i for i, p in enumerate(self.players) if not p.bankrupt]

        if len(active_players) == 1:
            self.done = True
            self.phase = "game_over"
            self.winner = active_players[0]
            self.last_message = f"Fim de jogo! {self.players[self.winner].name} venceu!"

    def _finish_by_net_worth(self):
        self.done = True
        self.phase = "game_over"
        self.winner = max(
            range(len(self.players)),
            key=lambda i: self._net_worth(i),
        )
        self.last_message = (
            f"Limite de turnos atingido. "
            f"{self.players[self.winner].name} venceu por patrimônio."
        )

    def _net_worth(self, player_index: int) -> int:
        player = self.players[player_index]
        if player.bankrupt:
            return -999999

        total = player.money

        for space in self.board:
            if space.type == "property" and space.owner == player_index:
                total += space.price

        return total

    def _is_valid_property_owner(self, prop_index: int, owner_index: int) -> bool:
        if prop_index < 0 or prop_index >= len(self.board):
            return False

        space = self.board[prop_index]
        return space.type == "property" and space.owner == owner_index

    def get_owned_properties(self, player_index: int) -> List[int]:
        return [
            i
            for i, space in enumerate(self.board)
            if space.type == "property" and space.owner == player_index
        ]

    def describe_pending_trade(self) -> List[str]:
        if not self.pending_trade:
            return []

        trade = self.pending_trade
        proposer = self.players[trade["from"]]
        target = self.players[trade["to"]]

        lines = [
            "Troca pendente:",
            f"{proposer.name} -> {target.name}",
            "",
            "Oferece:",
        ]

        if trade["offer_properties"]:
            for prop_index in trade["offer_properties"]:
                lines.append(f"- {self.board[prop_index].name}")
        else:
            lines.append("- Nenhuma propriedade")

        lines.append(f"- ${trade['offer_money']}")
        lines.append("")
        lines.append("Pede:")

        if trade["request_properties"]:
            for prop_index in trade["request_properties"]:
                lines.append(f"- {self.board[prop_index].name}")
        else:
            lines.append("- Nenhuma propriedade")

        lines.append(f"- ${trade['request_money']}")
        return lines

    def _sample_trade_actions(self, player_index: int, max_actions: int = 6) -> List[Dict[str, Any]]:
        actions = []
        proposer_properties = self.get_owned_properties(player_index)

        if not proposer_properties:
            return actions

        for target_index, target in enumerate(self.players):
            if target_index == player_index or target.bankrupt:
                continue

            target_properties = self.get_owned_properties(target_index)

            if not target_properties:
                continue

            for _ in range(2):
                offer_property = self.random.choice(proposer_properties)
                request_property = self.random.choice(target_properties)

                offer_money = self.random.choice([0, 50, 100, 150])
                request_money = self.random.choice([0, 50, 100])

                offer_money = min(offer_money, self.players[player_index].money)
                request_money = min(request_money, target.money)

                actions.append(
                    {
                        "type": PROPOSE_TRADE,
                        "target_player": target_index,
                        "offer_properties": [offer_property],
                        "offer_money": offer_money,
                        "request_properties": [request_property],
                        "request_money": request_money,
                    }
                )

                if len(actions) >= max_actions:
                    return actions

        return actions
