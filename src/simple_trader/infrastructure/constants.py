from enum import Enum, auto


class StrategyAction(Enum):
    NOTHING = auto()
    OPEN_LONG = auto()
    OPEN_SHORT = auto()
    CLOSE_LONG = auto()
    CLOSE_SHORT = auto()
    MOVE_STOP_LOSS_LONG = auto()
    MOVE_STOP_LOSS_SHORT = auto()
    DO_STOP_LOSS = auto()


class PositionState(Enum):
    WAIT = "WAIT"
    WAIT_BUY = "WAIT_BUY"
    WAIT_SELL = "WAIT_SELL"
    WAIT_SAFETY_BUY = "WAIT_SAFETY_BUY"
    WAIT_SAFETY_SELL = "WAIT_SAFETY_SELL"


class PositionType(Enum):
    UNKNOWN = "UNKNOWN"
    LONG = "LONG"
    SHORT = "SHORT"
