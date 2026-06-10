"""SignalChain — sequential state machine for composing signals (Phase 05, Task 01)."""

from signals_lib.base_signal import BaseSignal


class SignalChain:
    """Sequential state machine that advances through an ordered list of signals.

    A chain with N signals completes only after signal 0 fires on one tick,
    THEN signal 1 fires on a subsequent tick, and so on.  When all signals
    have fired the chain is considered complete and ``get_action()`` returns
    a trading action tuple.

    Args:
        name: Chain identifier used in log messages.
        res_action: ``STRATEGY_ACTION_*`` constant returned when the chain
            completes.
        tf: Timeframe (in minutes) this chain operates on.
        notify: When ``True`` (default), progress and completion events are
            logged via ``action.add_multiply_action()``.
    """

    def __init__(self, name: str, res_action: int, tf: int, notify: bool = True) -> None:
        self.name: str = name
        self.result_action = res_action
        self.tf: int = tf
        self.notify: bool = notify

        self.signals: list[BaseSignal] = []
        self.cur_pos: int = 0
        self.timer: float = 0
        self._completion_logged: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, signal: BaseSignal) -> None:
        """Append *signal* to the ordered signal sequence.

        Args:
            signal: A ``BaseSignal`` instance to add to the chain.
        """
        self.signals.append(signal)

    def check(self, data_point, levels: dict, cur_time: float, action) -> None:
        """Evaluate the current signal in the sequence.

        Only ``signals[cur_pos]`` is evaluated.  If it fires the chain
        advances to the next position (and ``timer`` is updated).  After
        advancing, if the new ``signals[cur_pos].reset()`` returns ``True``
        the entire chain is reset to position 0.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels passed through to the signal.
            cur_time: Unix timestamp of the current tick (used to set ``timer``).
            action: Action accumulator; may be ``None`` (e.g. in tests).
        """
        if self.cur_pos >= len(self.signals):
            return

        signal = self.signals[self.cur_pos]
        fired = signal.check(data_point, levels, action)

        if not fired:
            return

        # Log the firing event BEFORE advancing cur_pos (so log shows old pos)
        if action is not None and self.notify:
            marker = signal.get_marker_pos(data_point)
            action.add_multiply_action(
                marker,
                f"SF: C {self.name} S {self.cur_pos}",
            )

        self.cur_pos += 1
        self.timer = cur_time

        # If we haven't reached the end, check whether the next signal
        # immediately wants to abort the chain.
        if self.cur_pos < len(self.signals):
            if self.signals[self.cur_pos].reset():
                self.reset(action)

    def completed(self, data_point, action) -> bool:
        """Return ``True`` if all signals have fired.

        When the chain is complete and ``notify=True``, a completion message
        is logged via ``action.add_multiply_action()``.

        Args:
            data_point: Current market data point (used for marker position).
            action: Action accumulator; may be ``None``.

        Returns:
            ``True`` if ``cur_pos == len(signals)``, ``False`` otherwise.
        """
        if self.cur_pos == len(self.signals) and len(self.signals) > 0:
            if action is not None and self.notify and not self._completion_logged:
                self._completion_logged = True
                marker = self.signals[-1].get_marker_pos(data_point)
                action.add_multiply_action(marker, f"CPLTD: {self.name}")
            return True
        return False

    def get_action(self, price: float, action) -> list:
        """Return the action tuple for this completed chain.

        The data dict is built by merging ``{"position_time": tf}`` with the
        ``get_data()`` result from every signal in the chain (in order).

        Args:
            price: Current price to embed in the action tuple.
            action: Action accumulator (passed through; currently unused here).

        Returns:
            ``[result_action, tf, price, data_dict]``
        """
        assert self.cur_pos == len(self.signals), "get_action() called before chain completed"
        data_dict: dict = {"position_time": self.tf}
        for signal in self.signals:
            data_dict.update(signal.get_data())
        return [self.result_action, self.tf, price, data_dict]

    def reset(self, action) -> None:
        """Reset the chain to the initial state.

        Args:
            action: Action accumulator (unused; accepted for API consistency).
        """
        self.cur_pos = 0
        self.timer = 0
        self._completion_logged = False
