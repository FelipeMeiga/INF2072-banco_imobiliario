class Property:
    def __init__(self, name, price, rent, color_group=None):
        self.name = name
        self.price = price
        self.rent = rent
        self.color_group = color_group
        self.owner = None
        self.houses = 0
        self.mortgaged = False

    def is_owned(self):
        return self.owner is not None

    def calculate_rent(self):
        if self.mortgaged:
            return 0

        return self.rent * (1 + self.houses)