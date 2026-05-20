from typing import List
from .signal_chain import SignalChain
from ..data.data_point import DataPoint


class SignalManager:
    def __init__(self) -> None:
        self.chains: List[SignalChain] = []

    def add(self, chain: SignalChain) -> "SignalManager":
        self.chains.append(chain)
        return self

    def check(self, data_point: DataPoint) -> List[SignalChain]:
        completed: List[SignalChain] = []
        for chain in self.chains:
            if chain.check(data_point):
                completed.append(chain)
                chain.reset()
        return completed

    def reset_all(self) -> None:
        for chain in self.chains:
            chain.reset()
