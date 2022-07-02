from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from interlocking_cycles.cycle import InterlockingCycles


class Color(Enum):
    WHITE = auto()
    ORANGE = auto()
    GREEN = auto()
    RED = auto()
    YELLOW = auto()
    BLUE = auto()


class Face(Enum):
    U = UP = auto()
    D = DOWN = auto()
    R = RIGHT = auto()
    L = LEFT = auto()
    F = FRONT = auto()
    B = BACK = auto()

    @property
    def color(self) -> Color:
        return FACE_COLOR[self]


FACE_COLOR: dict[Face, Color] = {
    Face.UP: Color.BLUE,
    Face.DOWN: Color.GREEN,
    Face.RIGHT: Color.RED,
    Face.LEFT: Color.ORANGE,
    Face.FRONT: Color.WHITE,
    Face.BACK: Color.YELLOW,
}


@dataclass
class Corner:
    colors: tuple[Color, Color, Color]  # Clockwise direction, first color is vertical
    rotation: Literal[0, 1, 2]  # Number of clockwise turns past correct

    def rotate(self, face: Face, amount: int) -> None:
        if face in {Face.UP, Face.DOWN}:
            return
        elif face is Face.RIGHT:
            pass


@dataclass
class Edge:
    colors: tuple[Color, Color]
    rotation: Literal[0, 1]


@dataclass
class Center:
    color: Color


Piece = Corner | Edge | Center

# Cube coordinates have the x-axis to the right, y-axis upwards, and z-axis towards you
Coordinate = tuple[int, int, int]


class Cube2:
    def __init__(self) -> None:
        self.interlocking_cycles = InterlockingCycles[Face, Coordinate, Corner,](
            {
                Face.U: ((1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1)),
                Face.D: ((1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1)),
                Face.R: ((1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1)),
                Face.L: ((-1, 1, 1), (-1, -1, 1), (-1, -1, -1), (-1, 1, -1)),
                Face.F: ((1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1)),
                Face.B: ((1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)),
            },
            {
                (1, 1, 1): Corner((Color.WHITE, Color.RED, Color.BLUE), 0),
                (1, -1, 1): Corner((Color.WHITE, Color.RED, Color.GREEN), 0),
                (-1, -1, 1): Corner((Color.WHITE, Color.ORANGE, Color.GREEN), 0),
                (-1, 1, 1): Corner((Color.WHITE, Color.ORANGE, Color.BLUE), 0),
                (1, 1, -1): Corner((Color.YELLOW, Color.RED, Color.BLUE), 0),
                (1, -1, -1): Corner((Color.YELLOW, Color.RED, Color.GREEN), 0),
                (-1, -1, -1): Corner((Color.YELLOW, Color.ORANGE, Color.GREEN), 0),
                (-1, 1, -1): Corner((Color.YELLOW, Color.ORANGE, Color.BLUE), 0),
            },
        )

    def turn(self, face: Face, amount: int) -> None:
        self.interlocking_cycles.rotate(face, amount)
        for position in self.interlocking_cycles.cycles[face]:
            position.value.rotate(face, amount)


if __name__ == "__main__":
    from pprint import pprint

    c = Cube2()
    pprint(c.interlocking_cycles.cycles)
    c.turn(Face.R, -1)
    print("\n*" * 5)
    pprint(c.interlocking_cycles.cycles)
