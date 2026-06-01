import random
from typing import List

from game.models import Space

NUM_SPACES = 40


def create_board(seed: int | None = None) -> List[Space]:
    if seed is not None:
        random.seed(seed)

    board: List[Space] = []
    board.append(Space("Início", "start"))

    for i in range(1, NUM_SPACES):
        if i in [5, 15, 25, 35]:
            board.append(Space(f"Estação {i}", "property", price=200, rent=25))
        elif i in [4, 12, 28, 38]:
            board.append(Space(f"Imposto {i}", "tax", rent=100))
        elif i in [7, 22, 36]:
            board.append(Space(f"Sorte/Reves {i}", "chance"))
        elif i == 10:
            board.append(Space("Prisão", "jail"))
        elif i in [20, 30]:
            board.append(Space(f"Livre {i}", "empty"))
        else:
            price = random.choice([100, 120, 140, 160, 180, 200, 220, 240])
            rent = int(price * 0.12)
            board.append(Space(f"Rua {i}", "property", price=price, rent=rent))

    return board
