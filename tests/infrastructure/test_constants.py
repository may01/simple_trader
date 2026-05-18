from simple_trader.infrastructure.constants import StrategyAction, PositionState, PositionType

def test_strategy_action_nothing_exists():
    assert StrategyAction.NOTHING is not None

def test_strategy_action_open_long_exists():
    assert StrategyAction.OPEN_LONG is not None

def test_strategy_action_all_distinct():
    actions = [
        StrategyAction.NOTHING,
        StrategyAction.OPEN_LONG,
        StrategyAction.OPEN_SHORT,
        StrategyAction.CLOSE_LONG,
        StrategyAction.CLOSE_SHORT,
        StrategyAction.MOVE_STOP_LOSS_LONG,
        StrategyAction.MOVE_STOP_LOSS_SHORT,
        StrategyAction.DO_STOP_LOSS,
    ]
    assert len(set(actions)) == len(actions)

def test_position_state_wait_is_string_wait():
    assert PositionState.WAIT.value == "WAIT"

def test_position_state_all_states_exist():
    states = [
        PositionState.WAIT,
        PositionState.WAIT_BUY,
        PositionState.WAIT_SELL,
        PositionState.WAIT_SAFETY_BUY,
        PositionState.WAIT_SAFETY_SELL,
    ]
    assert len(states) == 5

def test_position_type_unknown_exists():
    assert PositionType.UNKNOWN is not None
