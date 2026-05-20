class Coin:
    def __init__(self, name: str, size: float = 0.0) -> None:
        self.name = name
        self.size = size
        self.used = 0.0
        self.loan = 0.0
        self.returned = 0.0
        self.want_to_use = 0.0
        self.action_amount = 0.0

    def use(self, amount: float) -> None:
        self.size -= amount
        self.used += amount

    def free(self, amount: float) -> None:
        self.used -= amount
        self.size += amount

    def borrow(self, amount: float) -> None:
        self.loan += amount
        self.size += amount

    def repay(self, amount: float) -> None:
        self.loan -= amount
        self.size -= amount

    @property
    def total(self) -> float:
        return self.size + self.used
