from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property
from typing import Literal

from interlocking_cycles.cycle import InterlockingCycles

# Cube coordinates have the x-axis to the right, y-axis upwards, and z-axis towards you
Coordinate = tuple[int, int, int]


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2


class Color(Enum):
    WHITE = auto()
    ORANGE = auto()
    GREEN = auto()
    RED = auto()
    YELLOW = auto()
    BLUE = auto()


class Face(Enum):
    UP = auto()
    DOWN = auto()
    RIGHT = auto()
    LEFT = auto()
    FRONT = auto()
    BACK = auto()

    @cached_property
    def color(self) -> Color:
        """The color of the face on the cube"""
        return {
            Face.UP: Color.BLUE,
            Face.DOWN: Color.GREEN,
            Face.RIGHT: Color.RED,
            Face.LEFT: Color.ORANGE,
            Face.FRONT: Color.WHITE,
            Face.BACK: Color.YELLOW,
        }[self]

    @cached_property
    def axis(self) -> Axis:
        """The axis the face rotates on"""
        if self is Face.RIGHT or self is Face.LEFT:
            return Axis.X
        elif self is Face.UP or self is Face.DOWN:
            return Axis.Y
        elif self is Face.FRONT or self is Face.BACK:
            return Axis.Z

    @cached_property
    def sign(self) -> int:
        """The sign of the non-zero coordinate of the face"""
        return 1 if self in {Face.UP, Face.RIGHT, Face.FRONT} else -1

    @cache
    def coordinate(self, cube_radius: int) -> Coordinate:
        """The coordinate of the face"""
        coordinate = [0] * 3
        coordinate[self.axis] = self.sign * cube_radius

        return tuple(coordinate)  # type: ignore

    @cache
    def rotate(self, axis: Axis, amount: int) -> Face:
        amount %= 4
        if amount == 0:
            return self

        if axis is Axis.X:
            new_face = {
                Face.UP: Face.BACK,
                Face.DOWN: Face.FRONT,
                Face.RIGHT: Face.RIGHT,
                Face.LEFT: Face.LEFT,
                Face.FRONT: Face.UP,
                Face.BACK: Face.DOWN,
            }[self]
        elif axis is Axis.Y:
            new_face = {
                Face.UP: Face.UP,
                Face.DOWN: Face.DOWN,
                Face.RIGHT: Face.FRONT,
                Face.LEFT: Face.BACK,
                Face.FRONT: Face.LEFT,
                Face.BACK: Face.RIGHT,
            }[self]
        elif axis is Axis.Z:
            new_face = {
                Face.UP: Face.RIGHT,
                Face.DOWN: Face.LEFT,
                Face.RIGHT: Face.DOWN,
                Face.LEFT: Face.UP,
                Face.FRONT: Face.FRONT,
                Face.BACK: Face.BACK,
            }[self]

        return new_face.rotate(axis, amount - 1)


@dataclass
class Corner:
    colors: tuple[Color, Color, Color]  # Clockwise direction, first color is vertical
    # rotation: Literal[0, 1, 2]  # Number of clockwise turns past correct
    facing: Face  # The face of the first color

    def rotate(self, face: Face, amount: int) -> None:
        self.facing = self.facing.rotate(face.axis, amount * face.sign)


@dataclass
class Edge:
    colors: tuple[Color, Color]
    rotation: Literal[0, 1]

    def rotate(self, face: Face, amount: int) -> None:
        raise NotImplementedError


@dataclass
class Center:
    color: Color

    def rotate(self, face: Face, amount: int) -> None:
        raise NotImplementedError


Piece = Corner | Edge | Center


def make_square_perimiter(
    radius: int, include_center: bool
) -> tuple[tuple[int, int], ...]:
    """Return the coordinates of the perimiter of a square with given radius"""
    edge = tuple(
        filter(
            lambda value: True if include_center else value != 0,
            range(radius, -radius, -1),
        )
    )

    return tuple(
        itertools.chain(
            ((radius, value) for value in edge),
            ((value, -radius) for value in edge),
            ((-radius, -value) for value in edge),
            ((-value, radius) for value in edge),
        )
    )


def make_cube_cycle(
    face: Face, axis_coordinate: int, cycle_radius: int, include_center: bool
) -> tuple[Coordinate, ...]:
    """Return a clockwise cycle on the face with given cycle radius"""

    # Cycle in positive direction of rotation wrt the axis (counter-clockwise)
    cycle_2d = make_square_perimiter(cycle_radius, include_center)

    cycle: tuple[Coordinate, ...] = tuple(
        (*point[: face.axis], axis_coordinate, *point[face.axis :])
        for point in cycle_2d
    )  # type: ignore

    # The negative sign of the face switches the order of the cycle, so it is clockwise
    if face.sign < 0:
        return cycle

    # Reverse the order on the positive face so the cycle is clockwise
    return tuple(reversed(cycle))


@dataclass(frozen=True)
class Cycle:
    face: Face
    offset: int  # Positive offset towards center
    is_slice: bool


class Cube:
    def __init__(self, size: int) -> None:
        self.size = size
        self.radius = size // 2
        self.amt_rings = (size + 1) // 2
        self.include_center = size % 2 == 1  # Include center on odd-sided cubes

        config: dict[Cycle, tuple[Coordinate, ...]] = {}
        values: dict[Coordinate, Piece] = {}

        # NOTE: Slices are duplicated
        for face in Face:
            for offset in range(1, self.size - 1):
                axis_coordinate = offset - self.radius
                if not self.include_center and axis_coordinate >= 0:
                    axis_coordinate += 1

                config[Cycle(face, offset, True)] = make_cube_cycle(
                    face,
                    axis_coordinate=axis_coordinate,
                    cycle_radius=self.radius,
                    include_center=self.include_center,
                )

        for face in Face:
            for offset in range(self.amt_rings):
                cycle = make_cube_cycle(
                    face,
                    axis_coordinate=face.sign * self.radius,
                    cycle_radius=self.radius - offset,
                    include_center=self.include_center,
                )

                config[Cycle(face, offset, False)] = cycle

                if offset > 0:  # Centers
                    values.update(
                        {coordinate: Center(face.color) for coordinate in cycle}
                    )

        # TODO: Edges
        for x, y, z in itertools.product(
            *((-self.radius, self.radius) for _ in range(3))
        ):
            x_face = Face.LEFT if x < 0 else Face.RIGHT
            y_face = Face.DOWN if y < 0 else Face.UP
            z_face = Face.BACK if z < 0 else Face.FRONT

            clockwise_sides = (Face.FRONT, Face.LEFT, Face.BACK, Face.RIGHT)

            x_index = clockwise_sides.index(x_face)
            x_first = clockwise_sides[(x_index + 1) % 4] is z_face

            if y_face is Face.DOWN:
                x_first = not x_first

            colors = (
                (y_face.color, x_face.color, z_face.color)
                if x_first
                else (y_face.color, z_face.color, x_face.color)
            )

            values[(x, y, z)] = Corner(colors, y_face)
            # print((x, y, z), values[(x, y, z)].colors, values[(x, y, z)].facing.name)

        self.interlocking_cycles = InterlockingCycles[Cycle, Coordinate, Piece](
            config, values
        )

    def turn(self, face: Face, amount: int, offset: int = 0) -> None:
        """Turn the face at the given offset the given amount"""
        assert 0 <= offset < self.size - 1

        if offset > 0:
            cycle_id = Cycle(face, offset, True)
            self.interlocking_cycles.rotate(cycle_id, -amount)
            for position in self.interlocking_cycles.cycles[cycle_id]:
                position.value.rotate(face, amount * (self.size - 1))
        else:
            for offset in range(self.amt_rings):
                cycle_id = Cycle(face, offset, False)
                self.interlocking_cycles.rotate(cycle_id, -amount)
                for position in self.interlocking_cycles.cycles[cycle_id]:
                    position.value.rotate(face, amount * (self.size - 1 - 2 * offset))
                    pass

    @property
    def layout(self) -> str:
        # Copied, almost
        edge = tuple(
            filter(
                lambda value: True if self.include_center else value != 0,
                range(-self.radius, self.radius + 1),
            )
        )
        square = tuple(itertools.product(edge, edge))
        result: list[list[str]] = [[" "] * self.size * 4 for _ in range(self.size * 3)]

        # Copied
        for face in Face:
            face_points: tuple[Coordinate, ...] = tuple(
                (*point[: face.axis], self.radius * face.sign, *point[face.axis :])
                for point in square
            )  # type: ignore

            # print(face)
            x_offset = {
                Face.LEFT: 0,
                Face.FRONT: 1,
                Face.RIGHT: 2,
                Face.BACK: 3,
                Face.UP: 1,
                Face.DOWN: 1,
            }[face]
            y_offset = {
                Face.LEFT: 1,
                Face.FRONT: 1,
                Face.RIGHT: 1,
                Face.BACK: 1,
                Face.UP: 2,  # Offset from bottom
                Face.DOWN: 0,
            }[face]

            for square_point, face_point in zip(square, face_points):
                x, y = square_point

                if face is Face.UP:
                    y *= -1
                elif face is Face.LEFT:
                    x, y = y, x
                elif face is Face.RIGHT:
                    x, y = y, x
                    x *= -1
                if face is Face.BACK:
                    x *= -1

                if not self.include_center:
                    if x > 0:
                        x -= 1
                    if y > 0:
                        y -= 1
                x += self.radius
                y += self.radius

                piece = self.interlocking_cycles.positions[face_point].value
                # print("\t", x, y, face_point, piece.colors, piece.facing.name)
                result[y + self.size * y_offset][
                    x + self.size * x_offset
                ] = piece.colors[0].name[0]
                result[y + self.size * y_offset][
                    x + self.size * x_offset
                ] = piece.facing.name[0]

        return "\n".join("".join(line) for line in reversed(result))


class Cube2:
    def __init__(self) -> None:
        self.interlocking_cycles = InterlockingCycles[Face, Coordinate, Corner](
            {
                Face.UP: ((1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1)),
                Face.DOWN: ((1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1)),
                Face.RIGHT: ((1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1)),
                Face.LEFT: ((-1, 1, 1), (-1, -1, 1), (-1, -1, -1), (-1, 1, -1)),
                Face.FRONT: ((1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1)),
                Face.BACK: ((1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)),
            },
            {
                (1, 1, 1): Corner((Color.WHITE, Color.RED, Color.BLUE), Face.UP),
                (1, -1, 1): Corner((Color.WHITE, Color.RED, Color.GREEN), Face.UP),
                (-1, -1, 1): Corner((Color.WHITE, Color.ORANGE, Color.GREEN), Face.UP),
                (-1, 1, 1): Corner((Color.WHITE, Color.ORANGE, Color.BLUE), Face.UP),
                (1, 1, -1): Corner((Color.YELLOW, Color.RED, Color.BLUE), Face.DOWN),
                (1, -1, -1): Corner((Color.YELLOW, Color.RED, Color.GREEN), Face.DOWN),
                (-1, -1, -1): Corner(
                    (Color.YELLOW, Color.ORANGE, Color.GREEN), Face.DOWN
                ),
                (-1, 1, -1): Corner(
                    (Color.YELLOW, Color.ORANGE, Color.BLUE), Face.DOWN
                ),
            },
        )

    def turn(self, face: Face, amount: int, offset: int = 0) -> None:
        """Turn the face at the given offset the given amount"""
        assert offset >= 0
        if offset > 0:
            self.interlocking_cycles.rotate(face, amount)
        for position in self.interlocking_cycles.cycles[face]:
            position.value.rotate(face, amount)


if __name__ == "__main__":
    from pprint import pprint

    """
    c = Cube2()
    pprint(c.interlocking_cycles.cycles)
    c.turn(Face.RIGHT, -1)
    print("\n*" * 5)
    pprint(c.interlocking_cycles.cycles)
    """
    c = Cube(2)
    # pprint(c.interlocking_cycles.cycles)
    print(c.layout)
    print("\n*" * 5)
    # pprint(c.interlocking_cycles.cycles)
    c.turn(Face.RIGHT, 1)
    # c.turn(Face.LEFT, -1)
    print(c.layout)

    print(Face.UP.rotate(Axis.X, 1))
