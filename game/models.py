from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Space:
    name: str
    type: str  # start, property, tax, chance, empty, jail
    price: int = 0
    rent: int = 0
    owner: Optional[int] = None

    # Mecânica de regiões/casas/hotel
    # group: país/região da propriedade. Só propriedades do mesmo grupo formam monopólio.
    # build_cost: preço para adicionar 1 casa nessa propriedade.
    # houses: nível de construção. 0 = sem casa, 1..4 = casas, 5 = hotel.
    # rent_schedule: aluguel por nível de construção, índices 0..5.
    group: Optional[str] = None
    build_cost: int = 0
    houses: int = 0
    rent_schedule: Tuple[int, int, int, int, int, int] = field(
        default_factory=lambda: (0, 0, 0, 0, 0, 0)
    )

    def current_rent(self) -> int:
        if self.type != "property":
            return self.rent

        if self.rent_schedule and 0 <= self.houses < len(self.rent_schedule):
            return self.rent_schedule[self.houses]

        return self.rent


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
