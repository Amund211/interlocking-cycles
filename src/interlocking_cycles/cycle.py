import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Hashable, Literal, TypeVar

CycleID = TypeVar("CycleID", bound=Hashable)
PositionID = TypeVar("PositionID", bound=Hashable)
ValueType = TypeVar("ValueType")


@dataclass
class Position(Generic[ValueType]):
    value: ValueType


def rotate(cycle: Sequence[Position[ValueType]], steps: int) -> None:
    """Rotate the cycle in place"""
    cycle_length = len(cycle)
    result = tuple(cycle[(i - steps) % cycle_length].value for i in range(cycle_length))
    for value, position in zip(result, cycle):
        position.value = value


class InterlockingCycles(Generic[CycleID, PositionID, ValueType]):
    def __init__(
        self,
        config: Mapping[CycleID, Sequence[PositionID]],
        values: Mapping[PositionID, ValueType],
    ):
        unique_position_ids = set(itertools.chain(*config.values()))
        self.positions = {
            position_id: Position(values[position_id])
            for position_id in unique_position_ids
        }

        self.cycles: dict[CycleID, tuple[Position[ValueType], ...]] = {
            cycle_id: tuple(self.positions[position_id] for position_id in position_ids)
            for cycle_id, position_ids in config.items()
        }

    def rotate(self, cycle_id: CycleID, steps: int) -> None:
        rotate(self.cycles[cycle_id], steps)


if __name__ == "__main__":
    config: dict[Literal[1, 2, 3, 4], tuple[int, ...]] = {
        1: (1, 2, 3, 4),
        2: (2, 3, 4, 5),
        3: (3, 4, 5, 6),
        4: (4, 5, 6, 7),
    }
    values: dict[int, int] = {i: i for i in range(1, 8)}

    ic = InterlockingCycles[Literal[1, 2, 3, 4], int, int](config, values)
    from pprint import pprint

    pprint(ic.cycles)

    ic.rotate(1, 1)
    pprint(ic.cycles)
