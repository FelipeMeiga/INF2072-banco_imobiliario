from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Space:
    name: str
    type: str  # start, property, tax, chance, empty, jail
    price: int = 0
    rent: int = 0
    owner: Optional[int] = None


@dataclass
class Player:
    name: str
    color: Tuple[int, int, int]
    money: int
    position: int = 0
    bankrupt: bool = False

    def move(self, steps: int, board_size: int) -> bool:
        old_position = self.position
        self.position = (self.position + steps) % board_size
        return old_position + steps >= board_size
