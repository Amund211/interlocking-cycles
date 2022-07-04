from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property
from typing import TypeVar

from interlocking_cycles.cycle import InterlockingCycles

T = TypeVar("T")


def insert_value(
    original: tuple[tuple[T, ...], ...], value: T, index: int
) -> tuple[tuple[T, ...], ...]:
    return tuple((*element[:index], value, *element[index:]) for element in original)


# Cube coordinates have the x-axis to the right, y-axis upwards, and z-axis towards you
Coordinate = tuple[int, int, int]


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2

    @cache
    def face(self, positive: bool) -> Face:
        """Get the positive/negative face on the axis"""
        if self is Axis.X:
            return Face.RIGHT if positive else Face.LEFT
        elif self is Axis.Y:
            return Face.UP if positive else Face.DOWN
        elif self is Axis.Z:
            return Face.FRONT if positive else Face.BACK

    @cached_property
    def side_faces(self) -> tuple[Face, Face, Face, Face]:
        """Get the side faces in clockwise order"""
        if self is Axis.X:
            return (Face.FRONT, Face.UP, Face.BACK, Face.DOWN)
        elif self is Axis.Y:
            return (Face.FRONT, Face.LEFT, Face.BACK, Face.RIGHT)
        elif self is Axis.Z:
            return (Face.UP, Face.RIGHT, Face.DOWN, Face.LEFT)


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
    def rotate(self, axis: Axis, amount: int) -> Face:
        """Rotate the face on the given axis the given amount"""
        if self.axis is axis:
            return self
        return axis.side_faces[(axis.side_faces.index(self) + amount) % 4]


@dataclass
class Corner:
    colors: tuple[Color, Color, Color]  # Clockwise direction
    facing: Face  # The face of the first color

    def rotate(self, face: Face, amount: int) -> None:
        self.facing = self.facing.rotate(face.axis, amount * face.sign)


@dataclass
class Edge:
    colors: tuple[Color, Color]
    facing: Face  # The face of the first color

    def rotate(self, face: Face, amount: int) -> None:
        self.facing = self.facing.rotate(face.axis, amount * face.sign)


@dataclass
class Center:
    color: Color

    def rotate(self, face: Face, amount: int) -> None:
        # TODO: Maybe consider rotations of the face
        pass


Piece = Corner | Edge | Center


def valid_coordinates(values: Iterable[int], include_center: bool) -> tuple[int, ...]:
    """Filter out 0 from values if not include_center"""
    if include_center:
        return tuple(values)
    return tuple(filter(lambda value: value != 0, values))


def make_square_perimiter(
    radius: int, include_center: bool
) -> tuple[tuple[int, int], ...]:
    """Return the clockwise coordinates of the perimiter of a square"""
    if radius == 0 and include_center:
        return ((0, 0),)

    edge = valid_coordinates(range(radius, -radius, -1), include_center)

    return tuple(
        itertools.chain(
            ((radius, value) for value in edge),  # right v
            ((value, -radius) for value in edge),  # down <
            ((-radius, -value) for value in edge),  # left ^
            ((-value, radius) for value in edge),  # up >
        )
    )


def make_cube_cycle(
    face: Face, axis_coordinate: int, cycle_radius: int, include_center: bool
) -> tuple[Coordinate, ...]:
    """Return a clockwise cycle on the face with given cycle radius"""

    # Clockwise cycle wrt the axis
    cycle_2d = make_square_perimiter(cycle_radius, include_center)

    cycle: tuple[Coordinate, ...] = insert_value(
        cycle_2d,
        axis_coordinate,
        face.axis,
    )  # type: ignore

    # Reverse the order on the negative face so the cycle is clockwise
    if face.sign < 0:
        return tuple(reversed(cycle))

    return cycle


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

        # Slice cycles
        # NOTE: Slices are duplicated
        for face in Face:
            for offset in range(1, self.size - 1):
                axis_coordinate = self.radius - offset
                if not self.include_center and axis_coordinate <= 0:
                    axis_coordinate -= 1

                config[Cycle(face, offset, True)] = make_cube_cycle(
                    face,
                    axis_coordinate=axis_coordinate * face.sign,
                    cycle_radius=self.radius,
                    include_center=self.include_center,
                )

        # Face cycles
        for face in Face:
            for offset in range(self.amt_rings):
                cycle = make_cube_cycle(
                    face,
                    axis_coordinate=face.sign * self.radius,
                    cycle_radius=self.radius - offset,
                    include_center=self.include_center,
                )

                config[Cycle(face, offset, False)] = cycle

                # Center-pieces
                if offset > 0:
                    values.update(
                        {coordinate: Center(face.color) for coordinate in cycle}
                    )

        # Edge-pieces
        all_axes = set(Axis)
        for axis in Axis:
            # The order does not matter because we iterate over the same values
            axis_1, axis_2 = all_axes - {axis}
            for first, last in itertools.product(
                (-self.radius, self.radius), (-self.radius, self.radius)
            ):
                coordinate = [0] * 3
                coordinate[axis_1] = first
                coordinate[axis_2] = last
                for value in range(-self.radius + 1, self.radius):
                    if value == 0 and not self.include_center:
                        continue
                    coordinate[axis] = value

                    face_1 = axis_1.face(first > 0)
                    face_2 = axis_2.face(last > 0)

                    values[tuple(coordinate)] = Edge(  # type: ignore
                        (face_1.color, face_2.color), face_1
                    )

        # Corner-pieces
        for x, y, z in itertools.product(
            *((-self.radius, self.radius) for _ in range(3))
        ):
            x_face = Axis.X.face(x > 0)
            y_face = Axis.Y.face(y > 0)
            z_face = Axis.Z.face(z > 0)

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

        self.interlocking_cycles = InterlockingCycles[Cycle, Coordinate, Piece](
            config, values
        )

    def turn(self, face: Face, amount: int, offset: int = 0) -> None:
        """Turn the face at the given offset the given amount"""
        assert 0 <= offset < self.size - 1

        if offset > 0:
            # Slice move
            # Rotate the cycle and the pieces
            cycle_id = Cycle(face, offset, True)
            self.interlocking_cycles.rotate(cycle_id, amount * (self.size - 1))
            for position in self.interlocking_cycles.cycles[cycle_id]:
                position.value.rotate(face, amount)
        else:
            # Face move
            # Rotate the cycle and the pieces for each ring on the face
            for offset in range(self.amt_rings):
                cycle_id = Cycle(face, offset, False)
                self.interlocking_cycles.rotate(
                    cycle_id, amount * (self.size - 1 - 2 * offset)
                )
                for position in self.interlocking_cycles.cycles[cycle_id]:
                    position.value.rotate(face, amount)

    def related_faces(self, coordinate: Coordinate) -> tuple[Face, ...]:
        """Return a tuple of faces touching the piece at the given coordinate"""
        related_faces: list[Face] = []
        for value, axis in zip(coordinate, Axis):
            if abs(value) == self.radius:
                related_faces.append(axis.face(value > 0))

        return tuple(related_faces)

    def color(self, coordinate: Coordinate, face: Face) -> Color:
        """Return the color of the cube at the given coordinate on the given face"""
        piece = self.interlocking_cycles.positions[coordinate].value
        related_faces = self.related_faces(coordinate)
        assert face in related_faces

        if isinstance(piece, Center):
            return piece.color
        elif isinstance(piece, Edge):
            return piece.colors[0] if face is piece.facing else piece.colors[1]
        elif isinstance(piece, Corner):
            if face is piece.facing:
                return piece.colors[0]

            positive_x = coordinate[0] > 0
            x_face = Axis.X.face(positive_x)

            if (
                Axis.X.side_faces[
                    (
                        Axis.X.side_faces.index(related_faces[1])
                        + (1 if positive_x else -1)
                    )
                    % 4
                ]
                is related_faces[2]
            ):
                face_order = (x_face, related_faces[1], related_faces[2])
            else:
                face_order = (x_face, related_faces[2], related_faces[1])

            if face_order[(face_order.index(piece.facing) + 1) % 3] is face:
                return piece.colors[1]

            return piece.colors[2]

    @property
    def layout(self) -> str:
        """Return a string representation of the layout of the cube"""
        edge = valid_coordinates(
            range(-self.radius, self.radius + 1), self.include_center
        )
        square = tuple(itertools.product(edge, edge))
        result: list[list[str]] = [[" "] * self.size * 4 for _ in range(self.size * 3)]

        for face in Face:
            # Copied
            face_coordinates: tuple[Coordinate, ...] = insert_value(
                square,
                self.radius * face.sign,
                face.axis,
            )  # type: ignore

            # Offset from left edge
            x_offset = {
                Face.LEFT: 0,
                Face.FRONT: 1,
                Face.RIGHT: 2,
                Face.BACK: 3,
                Face.UP: 1,
                Face.DOWN: 1,
            }[face]

            # Offset from bottom edge
            y_offset = {
                Face.LEFT: 1,
                Face.FRONT: 1,
                Face.RIGHT: 1,
                Face.BACK: 1,
                Face.UP: 2,
                Face.DOWN: 0,
            }[face]

            for point_2d, coordinate in zip(square, face_coordinates):
                x, y = point_2d

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

                result[y + self.size * y_offset][x + self.size * x_offset] = self.color(
                    coordinate, face
                ).name[0]

        return "\n".join("".join(line) for line in reversed(result))


if __name__ == "__main__":

    def main() -> None:
        """
        from pprint import pprint
        c = Cube2()
        pprint(c.interlocking_cycles.cycles)
        c.turn(Face.RIGHT, -1)
        print("\n*" * 5)
        pprint(c.interlocking_cycles.cycles)
        """

        c = Cube(3)
        print(c.layout, "\n")
        c.turn(Face.RIGHT, 1)
        c.turn(Face.LEFT, -1)
        print(c.layout, "\n")

        c = Cube(15)
        print(c.layout, "\n")
        for offset in range(5):
            c.turn(Face.RIGHT, 2, offset)
            c.turn(Face.LEFT, -2, offset)
            c.turn(Face.UP, 2, offset)
            c.turn(Face.DOWN, -2, offset)
            c.turn(Face.FRONT, 2, offset)
            c.turn(Face.BACK, -2, offset)
        print(c.layout, "\n")

        c = Cube(4)
        c.turn(Face.UP, 2, 2)
        print(c.layout, "\n")
        c.turn(Face.RIGHT, 2, 1)
        print(c.layout, "\n")
        for offset in range(1, 2):
            c.turn(Face.RIGHT, 2, offset)
            c.turn(Face.LEFT, -2, offset)
            c.turn(Face.UP, 2, offset)
            c.turn(Face.DOWN, -2, offset)
            c.turn(Face.FRONT, 2, offset)
            c.turn(Face.BACK, -2, offset)
        print(c.layout, "\n")

    def timing(cube_size: int) -> tuple[float, float, float]:
        import time

        runs = max(int(16000 * cube_size ** (-2)), 1)
        start = time.perf_counter()
        for i in range(runs):
            Cube(cube_size)
        end = time.perf_counter()
        time_per_init = (end - start) / runs

        runs = max(int(20000 * cube_size ** (-2)), 1)
        c = Cube(cube_size)
        start = time.perf_counter()
        for i in range(runs):
            for face in Face:
                c.turn(face, 1)
        end = time.perf_counter()
        time_per_faceturn = (end - start) / runs / 6

        runs = max(int(20000 * cube_size ** (-1)), 1)
        c = Cube(cube_size)
        start = time.perf_counter()
        for i in range(runs):
            for face in Face:
                c.turn(face, 1, 1)
        end = time.perf_counter()
        time_per_sliceturn = (end - start) / runs / 6

        return time_per_init, time_per_faceturn, time_per_sliceturn

    def plot_timing() -> None:
        import math

        import matplotlib.pyplot as plt

        init = []
        faceturn = []
        sliceturn = []
        cube_sizes = tuple(range(3, 201))
        for cube_size in cube_sizes:
            time_per_init, time_per_faceturn, time_per_sliceturn = timing(cube_size)
            print("Cube size:", cube_size)
            print(f"\ttpi ={time_per_init:.2e} ips={1/time_per_init:.0f}")
            print(f"\ttpft={time_per_faceturn:.2e} tps={1/time_per_faceturn:.0f}")
            print(f"\ttpst={time_per_sliceturn:.2e} tps={1/time_per_sliceturn:.0f}")
            init.append(time_per_init)
            faceturn.append(time_per_faceturn)
            sliceturn.append(time_per_sliceturn)

        distance = math.log(cube_sizes[-1]) - math.log(cube_sizes[40])
        init_distance = math.log(init[-1]) - math.log(init[40])
        faceturn_distance = math.log(faceturn[-1]) - math.log(faceturn[40])
        sliceturn_distance = math.log(sliceturn[-1]) - math.log(sliceturn[40])
        print(init_distance / distance)
        print(faceturn_distance / distance)
        print(sliceturn_distance / distance)

        plt.loglog(cube_sizes, init, label="init")
        plt.loglog(cube_sizes, faceturn, label="faceturn")
        plt.loglog(cube_sizes, sliceturn, label="sliceturn")
        plt.grid()
        plt.legend()
        plt.show()

    main()

    plot_timing()
