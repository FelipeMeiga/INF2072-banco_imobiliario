from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Space:
    name: str
    type: str  # go, property, railroad, utility, tax, chance, community_chest, jail, free_parking, go_to_jail
    price: int = 0
    rent: int = 0
    owner: Optional[int] = None
    mortgaged: bool = False
    property_kind: str = "none"  # color, railroad, utility, none

    # group: color/set used to form a monopoly.
    # houses: 0 = empty, 1..4 = houses, 5 = hotel.
    group: Optional[str] = None
    build_cost: int = 0
    houses: int = 0
    rent_schedule: Tuple[int, int, int, int, int, int] = field(
        default_factory=lambda: (0, 0, 0, 0, 0, 0)
    )
    mortgage_value: int = 0
    tax_amount: int = 0

    def is_ownable(self) -> bool:
        return self.type in ("property", "railroad", "utility")

    def current_rent(self) -> int:
        if not self.is_ownable():
            return self.rent

        if self.mortgaged:
            return 0

        if self.type == "property" and self.rent_schedule and 0 <= self.houses < len(self.rent_schedule):
            return self.rent_schedule[self.houses]

        return self.rent


@dataclass
class Player:
    name: str
    color: Tuple[int, int, int]
    money: int
    position: int = 0
    bankrupt: bool = False
    in_jail: bool = False
    jail_turns: int = 0
    jail_free_cards: int = 0

    def move(self, steps: int, board_size: int) -> bool:
        old_position = self.position
        self.position = (self.position + steps) % board_size
        return old_position + steps >= board_size
