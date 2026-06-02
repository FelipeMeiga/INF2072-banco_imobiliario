import random
from typing import Any, Dict, List, Optional

from game.actions import (
    ACCEPT_TRADE,
    BUY_PROPERTY,
    DECLINE_TRADE,
    PASS_BUY,
    PROPOSE_TRADE,
    ROLL_DICE,
    BUILD_HOUSE,
)
from game.board import GROUPS, MAX_HOUSES, NUM_SPACES, create_board
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
    """
    Ambiente principal do jogo.

    Essa classe não depende de Pygame. Ela só sabe aplicar regras.
    A visualização e os agentes conversam com ela usando ações estruturadas.
    """

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
        self.action_count = 0
        self.last_message = "Jogo iniciado."
        self.last_action: Optional[Dict[str, Any]] = None
        self.event_history: List[Dict[str, Any]] = []
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
            "property_houses": [
                space.houses if space.type == "property" else 0
                for space in self.board
            ],
            "property_groups": [
                space.group if space.type == "property" else None
                for space in self.board
            ],
            "pending_trade": self.pending_trade,
            "done": self.done,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "action_count": self.action_count,
            "last_message": self.last_message,
            "event_history": self.event_history.copy(),
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

            # Construções entram como ações diretas do ambiente.
            # A IA não usa menu: ela escolhe construir em uma propriedade específica.
            actions.extend(self._get_build_actions(player_index))

            # Para não explodir o espaço de ações, geramos algumas propostas candidatas.
            # O agente aleatório pode escolher uma delas ou simplesmente rolar os dados.
            if not self.trade_proposed_this_turn:
                actions.extend(self._sample_trade_actions(player_index, max_actions=16))

            return actions

        return []

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.get_state(), 0.0, self.done, {"message": self.last_message}

        self.last_action = action
        acting_player_index = self._get_acting_player_for_action(action)
        before_net_worth = self._net_worth(acting_player_index)
        before_completed_groups = self._completed_group_count(acting_player_index)
        before_total_houses = self._total_houses(acting_player_index)
        action_type = action.get("type")

        if action_type == ROLL_DICE:
            self._roll_dice()
        elif action_type == BUY_PROPERTY:
            self._buy_property()
        elif action_type == PASS_BUY:
            self._pass_buy()
        elif action_type == PROPOSE_TRADE:
            self._propose_trade(action)
        elif action_type == BUILD_HOUSE:
            self._build_house(action)
        elif action_type == ACCEPT_TRADE:
            self._accept_trade()
        elif action_type == DECLINE_TRADE:
            self._decline_trade()
        else:
            self.last_message = f"Ação inválida: {action_type}"

        self._check_game_end()

        after_net_worth = self._net_worth(acting_player_index)
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

        if self.done and self.winner is not None:
            reward += 100.0 if self.winner == acting_player_index else -100.0

        self._register_event(acting_player_index, action, reward)

        return self.get_state(), reward, self.done, {"message": self.last_message}

    def _register_event(self, acting_player_index: int, action: Dict[str, Any], reward: float):
        self.action_count += 1

        player_name = self.players[acting_player_index].name
        action_type = action.get("type", "unknown")

        event = {
            "number": self.action_count,
            "turn": self.turn_count,
            "player": player_name,
            "action_type": action_type,
            "message": self.last_message,
            "reward": reward,
        }

        self.event_history.append(event)

        # Mantém só os últimos eventos para a interface não ficar pesada.
        if len(self.event_history) > 12:
            self.event_history = self.event_history[-12:]


    def _get_acting_player_for_action(self, action: Dict[str, Any]) -> int:
        if self.phase == "pending_trade_response" and self.pending_trade is not None:
            return self.pending_trade["to"]
        return self.current_player_index

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
            rent = self.get_rent_for_space(space)
            player.money -= rent
            owner.money += rent

            construction_text = self._construction_label(space)
            self.last_message = (
                f"{player.name} pagou ${rent} de aluguel para "
                f"{owner.name} por {space.name}{construction_text}."
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
            group_text = f" ({space.group})" if space.group else ""
            self.last_message = f"{player.name} comprou {space.name}{group_text} por ${space.price}."
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
    # CONSTRUÇÕES / CASAS / HOTEL
    # ========================================================

    def _build_house(self, action: Dict[str, Any]):
        if self.phase != "ready_to_roll":
            self.last_message = "Construções só podem ser feitas antes de rolar os dados."
            return

        prop_index = int(action.get("property_index", -1))

        if prop_index < 0 or prop_index >= len(self.board):
            self.last_message = "Propriedade inválida para construção."
            return

        space = self.board[prop_index]
        player = self.current_player

        if space.type != "property" or space.group is None:
            self.last_message = "Essa casa não aceita construções."
            return

        if space.owner != self.current_player_index:
            self.last_message = f"{player.name} não é dono de {space.name}."
            return

        if not self._owns_complete_group(self.current_player_index, space.group):
            self.last_message = (
                f"{player.name} precisa possuir todas as propriedades da região "
                f"{space.group} para construir."
            )
            return

        if space.houses >= MAX_HOUSES:
            self.last_message = f"{space.name} já tem hotel."
            return

        if player.money < space.build_cost:
            self.last_message = f"{player.name} não tem dinheiro para construir em {space.name}."
            return

        player.money -= space.build_cost
        space.houses += 1

        if space.houses >= MAX_HOUSES:
            level_text = "hotel"
        elif space.houses == 1:
            level_text = "1 casa"
        else:
            level_text = f"{space.houses} casas"

        new_rent = space.current_rent()
        self.last_message = (
            f"{player.name} construiu em {space.name}: {level_text}. "
            f"Novo aluguel: ${new_rent}."
        )

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
                space.houses = 0

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

        if action_type == BUY_PROPERTY:
            reward += 2.0

        if action_type == BUILD_HOUSE and after_total_houses > before_total_houses:
            reward += 4.0

        if action_type in (BUY_PROPERTY, ACCEPT_TRADE):
            completed_delta = after_completed_groups - before_completed_groups
            if completed_delta > 0:
                reward += 12.0 * completed_delta

        return reward

    def _completed_group_count(self, player_index: int) -> int:
        return sum(
            1
            for group in GROUPS
            if self._owns_complete_group(player_index, group)
        )

    def _total_houses(self, player_index: int) -> int:
        return sum(
            space.houses
            for space in self.board
            if space.type == "property" and space.owner == player_index
        )

    def _is_valid_property_owner(self, prop_index: int, owner_index: int) -> bool:
        if prop_index < 0 or prop_index >= len(self.board):
            return False

        space = self.board[prop_index]
        return space.type == "property" and space.owner == owner_index


    def _owns_complete_group(self, player_index: int, group: Optional[str]) -> bool:
        if group is None or group not in GROUPS:
            return False

        return all(
            self.board[prop_index].owner == player_index
            for prop_index in GROUPS[group]
        )

    def _get_build_actions(self, player_index: int) -> List[Dict[str, Any]]:
        player = self.players[player_index]

        if player.bankrupt:
            return []

        actions = []

        for prop_index, space in enumerate(self.board):
            if space.type != "property":
                continue

            if space.group is None:
                continue

            if space.owner != player_index:
                continue

            if space.houses >= MAX_HOUSES:
                continue

            if player.money < space.build_cost:
                continue

            if not self._owns_complete_group(player_index, space.group):
                continue

            actions.append({"type": BUILD_HOUSE, "property_index": prop_index})

        return actions

    def get_rent_for_space(self, space: Space) -> int:
        if space.type != "property":
            return space.rent

        rent = space.current_rent()

        # Bônus de monopólio: se o jogador possui toda a região, o aluguel
        # sem casas dobra. Com casas/hotel, o rent_schedule já representa o aumento.
        if (
            space.owner is not None
            and space.houses == 0
            and space.group is not None
            and self._owns_complete_group(space.owner, space.group)
        ):
            return rent * 2

        return rent

    def _construction_label(self, space: Space) -> str:
        if space.type != "property" or space.houses <= 0:
            return ""

        if space.houses >= MAX_HOUSES:
            return " com hotel"

        if space.houses == 1:
            return " com 1 casa"

        return f" com {space.houses} casas"

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

        seen_actions = set()

        def add_action(action: Dict[str, Any]) -> bool:
            key = (
                action["target_player"],
                tuple(sorted(action.get("offer_properties", []))),
                tuple(sorted(action.get("request_properties", []))),
                int(action.get("offer_money", 0)),
                int(action.get("request_money", 0)),
            )
            if key in seen_actions:
                return False

            seen_actions.add(key)
            actions.append(action)
            return len(actions) >= max_actions

        for action in self._get_region_completion_trade_actions(player_index):
            if add_action(action):
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

                if add_action(
                    {
                        "type": PROPOSE_TRADE,
                        "target_player": target_index,
                        "offer_properties": [offer_property],
                        "offer_money": offer_money,
                        "request_properties": [request_property],
                        "request_money": request_money,
                    }
                ):
                    return actions

        return actions

    def _get_region_completion_trade_actions(self, player_index: int) -> List[Dict[str, Any]]:
        player = self.players[player_index]
        actions = []

        for group, group_properties in GROUPS.items():
            owned_in_group = [
                prop_index
                for prop_index in group_properties
                if self.board[prop_index].owner == player_index
            ]
            missing_owned_by_others = [
                prop_index
                for prop_index in group_properties
                if self.board[prop_index].owner is not None
                and self.board[prop_index].owner != player_index
            ]

            if not owned_in_group or not missing_owned_by_others:
                continue

            missing_count = len(group_properties) - len(owned_in_group)
            if missing_count > 2:
                continue

            for request_property in missing_owned_by_others:
                target_index = self.board[request_property].owner
                target = self.players[target_index]

                if target.bankrupt:
                    continue

                offer_property = self._choose_trade_offer_property(
                    player_index,
                    target_index,
                    protected_group=group,
                )
                offer_properties = [] if offer_property is None else [offer_property]
                offer_money = self._trade_offer_money(player_index, request_property, missing_count)

                actions.append(
                    {
                        "type": PROPOSE_TRADE,
                        "target_player": target_index,
                        "offer_properties": offer_properties,
                        "offer_money": offer_money,
                        "request_properties": [request_property],
                        "request_money": 0,
                    }
                )

        return actions

    def _choose_trade_offer_property(
        self,
        player_index: int,
        target_index: int,
        protected_group: str,
    ) -> Optional[int]:
        candidates = [
            prop_index
            for prop_index in self.get_owned_properties(player_index)
            if self.board[prop_index].group != protected_group
        ]

        if not candidates:
            return None

        def offer_score(prop_index: int) -> tuple[int, int, int]:
            space = self.board[prop_index]
            target_group_count = 0

            if space.group in GROUPS:
                target_group_count = sum(
                    1
                    for group_prop_index in GROUPS[space.group]
                    if self.board[group_prop_index].owner == target_index
                )

            return target_group_count, -space.price, -prop_index

        return max(candidates, key=offer_score)

    def _trade_offer_money(
        self,
        player_index: int,
        request_property: int,
        missing_count: int,
    ) -> int:
        player = self.players[player_index]
        requested_space = self.board[request_property]
        target_offer = requested_space.price if missing_count == 1 else requested_space.price // 2

        return max(0, min(player.money, target_offer))
