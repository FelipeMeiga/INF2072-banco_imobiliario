from typing import Dict, List, Optional

from game.models import Space

NUM_SPACES = 40
MAX_HOUSES = 5

# Cada grupo representa um país/região. Ao possuir todas as propriedades do grupo,
# o jogador pode construir até 5 níveis naquela propriedade.
# 1..4 = casas, 5 = hotel.
GROUPS: Dict[str, List[int]] = {
    "Brasil": [1, 2, 3],
    "Argentina": [6, 8, 9],
    "Chile": [11, 13, 14],
    "Uruguai": [16, 17, 18, 19],
    "Portugal": [21, 23, 24],
    "Espanha": [26, 27, 29],
    "França": [31, 32, 33],
    "Itália": [34, 37, 39],
}

GROUP_PRICES = {
    "Brasil": 100,
    "Argentina": 120,
    "Chile": 140,
    "Uruguai": 160,
    "Portugal": 180,
    "Espanha": 200,
    "França": 220,
    "Itália": 240,
}

GROUP_BUILD_COSTS = {
    "Brasil": 50,
    "Argentina": 50,
    "Chile": 75,
    "Uruguai": 75,
    "Portugal": 100,
    "Espanha": 100,
    "França": 125,
    "Itália": 150,
}

SPACE_TO_GROUP: Dict[int, str] = {
    space_index: group
    for group, spaces in GROUPS.items()
    for space_index in spaces
}


def _rent_schedule(base_rent: int) -> tuple[int, int, int, int, int, int]:
    """
    Aluguel por nível de construção.

    0 = sem casa
    1..4 = casas
    5 = hotel

    Os multiplicadores são agressivos de propósito para as partidas terminarem
    com mais frequência por falência, e não por limite de turnos.
    """
    return (
        base_rent,
        base_rent * 5,
        base_rent * 10,
        base_rent * 20,
        base_rent * 35,
        base_rent * 60,
    )


def _create_group_property(index: int, group: str) -> Space:
    price = GROUP_PRICES[group]
    build_cost = GROUP_BUILD_COSTS[group]

    # Pequena variação para ruas do mesmo grupo não ficarem 100% iguais.
    group_spaces = GROUPS[group]
    offset = group_spaces.index(index)
    price += offset * 10

    base_rent = max(10, int(price * 0.12))

    return Space(
        name=f"{group} {offset + 1}",
        type="property",
        price=price,
        rent=base_rent,
        group=group,
        build_cost=build_cost,
        houses=0,
        rent_schedule=_rent_schedule(base_rent),
    )


def create_board(seed: Optional[int] = None) -> List[Space]:
    # O parâmetro seed fica por compatibilidade, mas o tabuleiro agora é determinístico.
    board: List[Space] = []
    board.append(Space("Início", "start"))

    for i in range(1, NUM_SPACES):
        if i in SPACE_TO_GROUP:
            board.append(_create_group_property(i, SPACE_TO_GROUP[i]))
        elif i in [5, 15, 25, 35]:
            board.append(
                Space(
                    f"Estação {i}",
                    "property",
                    price=200,
                    rent=25,
                    group=None,
                    build_cost=0,
                    houses=0,
                    rent_schedule=(25, 25, 25, 25, 25, 25),
                )
            )
        elif i in [4, 12, 28, 38]:
            board.append(Space(f"Imposto {i}", "tax", rent=100))
        elif i in [7, 22, 36]:
            board.append(Space(f"Sorte/Reves {i}", "chance"))
        elif i == 10:
            board.append(Space("Prisão", "jail"))
        elif i in [20, 30]:
            board.append(Space(f"Livre {i}", "empty"))
        else:
            board.append(Space(f"Livre {i}", "empty"))

    return board
