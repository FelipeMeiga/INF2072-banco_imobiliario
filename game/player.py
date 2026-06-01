class Player:
    def __init__(self, name, agent, money=1500, color=(0, 0, 0)):
        self.name = name
        self.agent = agent
        self.money = money
        self.position = 0
        self.properties = []
        self.is_bankrupt = False
        self.color = color

    def move(self, steps, board_size):
        old_position = self.position
        self.position = (self.position + steps) % board_size

        if self.position < old_position:
            self.receive(200)

    def pay(self, amount):
        self.money -= amount

        if self.money < 0:
            self.is_bankrupt = True

    def receive(self, amount):
        self.money += amount

    def add_property(self, property_):
        self.properties.append(property_)
        property_.owner = self

    def net_worth(self):
        return self.money + sum(p.price for p in self.properties)