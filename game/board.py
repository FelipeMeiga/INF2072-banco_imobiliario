from typing import Dict, List, Optional

from game.models import Space

NUM_SPACES = 40
MAX_HOUSES = 5
BANK_HOUSES = 32
BANK_HOTELS = 12
GO_INDEX = 0
JAIL_INDEX = 10
FREE_PARKING_INDEX = 20
GO_TO_JAIL_INDEX = 30

GROUPS: Dict[str, List[int]] = {
    "Brown": [1, 3],
    "Light Blue": [6, 8, 9],
    "Pink": [11, 13, 14],
    "Orange": [16, 18, 19],
    "Red": [21, 23, 24],
    "Yellow": [26, 27, 29],
    "Green": [31, 32, 34],
    "Dark Blue": [37, 39],
}

RAILROADS = [5, 15, 25, 35]
UTILITIES = [12, 28]

SPACE_TO_GROUP: Dict[int, str] = {
    space_index: group
    for group, spaces in GROUPS.items()
    for space_index in spaces
}


def _ownable(
    name: str,
    space_type: str,
    price: int,
    rent: int,
    group: Optional[str] = None,
    build_cost: int = 0,
    rent_schedule: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
) -> Space:
    return Space(
        name=name,
        type=space_type,
        price=price,
        rent=rent,
        group=group,
        build_cost=build_cost,
        rent_schedule=rent_schedule,
        mortgage_value=price // 2,
        property_kind="color" if space_type == "property" else space_type,
    )


def _property(
    name: str,
    price: int,
    rent_schedule: tuple[int, int, int, int, int, int],
    group: str,
    build_cost: int,
) -> Space:
    return _ownable(
        name=name,
        space_type="property",
        price=price,
        rent=rent_schedule[0],
        group=group,
        build_cost=build_cost,
        rent_schedule=rent_schedule,
    )


def _railroad(name: str) -> Space:
    return _ownable(
        name=name,
        space_type="railroad",
        price=200,
        rent=25,
        rent_schedule=(25, 25, 25, 25, 25, 25),
    )


def _utility(name: str) -> Space:
    return _ownable(
        name=name,
        space_type="utility",
        price=150,
        rent=0,
        rent_schedule=(0, 0, 0, 0, 0, 0),
    )


def create_board(seed: Optional[int] = None) -> List[Space]:
    # seed is kept for API compatibility. The classic board is deterministic.
    return [
        Space("Go", "go"),
        _property("Brown 1", 60, (2, 10, 30, 90, 160, 250), "Brown", 50),
        Space("Community Chest 1", "community_chest"),
        _property("Brown 2", 60, (4, 20, 60, 180, 320, 450), "Brown", 50),
        Space("Income Tax", "tax", tax_amount=200),
        _railroad("Railroad 1"),
        _property("Light Blue 1", 100, (6, 30, 90, 270, 400, 550), "Light Blue", 50),
        Space("Chance 1", "chance"),
        _property("Light Blue 2", 100, (6, 30, 90, 270, 400, 550), "Light Blue", 50),
        _property("Light Blue 3", 120, (8, 40, 100, 300, 450, 600), "Light Blue", 50),
        Space("Jail / Just Visiting", "jail"),
        _property("Pink 1", 140, (10, 50, 150, 450, 625, 750), "Pink", 100),
        _utility("Utility 1"),
        _property("Pink 2", 140, (10, 50, 150, 450, 625, 750), "Pink", 100),
        _property("Pink 3", 160, (12, 60, 180, 500, 700, 900), "Pink", 100),
        _railroad("Railroad 2"),
        _property("Orange 1", 180, (14, 70, 200, 550, 750, 950), "Orange", 100),
        Space("Community Chest 2", "community_chest"),
        _property("Orange 2", 180, (14, 70, 200, 550, 750, 950), "Orange", 100),
        _property("Orange 3", 200, (16, 80, 220, 600, 800, 1000), "Orange", 100),
        Space("Free Parking", "free_parking"),
        _property("Red 1", 220, (18, 90, 250, 700, 875, 1050), "Red", 150),
        Space("Chance 2", "chance"),
        _property("Red 2", 220, (18, 90, 250, 700, 875, 1050), "Red", 150),
        _property("Red 3", 240, (20, 100, 300, 750, 925, 1100), "Red", 150),
        _railroad("Railroad 3"),
        _property("Yellow 1", 260, (22, 110, 330, 800, 975, 1150), "Yellow", 150),
        _property("Yellow 2", 260, (22, 110, 330, 800, 975, 1150), "Yellow", 150),
        _utility("Utility 2"),
        _property("Yellow 3", 280, (24, 120, 360, 850, 1025, 1200), "Yellow", 150),
        Space("Go To Jail", "go_to_jail"),
        _property("Green 1", 300, (26, 130, 390, 900, 1100, 1275), "Green", 200),
        _property("Green 2", 300, (26, 130, 390, 900, 1100, 1275), "Green", 200),
        Space("Community Chest 3", "community_chest"),
        _property("Green 3", 320, (28, 150, 450, 1000, 1200, 1400), "Green", 200),
        _railroad("Railroad 4"),
        Space("Chance 3", "chance"),
        _property("Dark Blue 1", 350, (35, 175, 500, 1100, 1300, 1500), "Dark Blue", 200),
        Space("Luxury Tax", "tax", tax_amount=100),
        _property("Dark Blue 2", 400, (50, 200, 600, 1400, 1700, 2000), "Dark Blue", 200),
    ]
