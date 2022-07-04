from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property

from interlocking_cycles.cycle import InterlockingCycles

# Cube coordinates have the x-axis to the right, y-axis upwards, and z-axis towards you
Coordinate = tuple[int, int, int]


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2

    @cache
    def face(self, positive: bool) -> Face:
        if self is Axis.X:
            return Face.RIGHT if positive else Face.LEFT
        elif self is Axis.Y:
            return Face.UP if positive else Face.DOWN
        elif self is Axis.Z:
            return Face.FRONT if positive else Face.BACK


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
    facing: Face

    def rotate(self, face: Face, amount: int) -> None:
        self.facing = self.facing.rotate(face.axis, amount * face.sign)


@dataclass
class Center:
    color: Color

    def rotate(self, face: Face, amount: int) -> None:
        # TODO: Maybe consider rotations of the face
        pass


Piece = Corner | Edge | Center


def make_square_perimiter(
    radius: int, include_center: bool
) -> tuple[tuple[int, int], ...]:
    """Return the coordinates of the perimiter of a square with given radius"""
    if radius == 0 and include_center:
        return ((0, 0),)

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
        (*point[: face.axis], axis_coordinate, *point[face.axis :])  # type: ignore
        for point in cycle_2d
    )

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

        # Slice cycles
        # NOTE: Slices are duplicated
        for face in Face:
            for offset in range(1, self.size - 1):
                axis_coordinate = self.radius - offset
                if not self.include_center and axis_coordinate >= 0:
                    axis_coordinate += 1

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
            cycle_id = Cycle(face, offset, True)
            self.interlocking_cycles.rotate(cycle_id, -amount * (self.size - 1))
            for position in self.interlocking_cycles.cycles[cycle_id]:
                position.value.rotate(face, amount)
        else:
            for offset in range(self.amt_rings):
                cycle_id = Cycle(face, offset, False)
                self.interlocking_cycles.rotate(
                    cycle_id, -amount * (self.size - 1 - 2 * offset)
                )
                for position in self.interlocking_cycles.cycles[cycle_id]:
                    position.value.rotate(face, amount)
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
                (
                    *point[: face.axis],
                    self.radius * face.sign,
                    *point[face.axis :],
                )  # type: ignore
                for point in square
            )

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

                result[y + self.size * y_offset][x + self.size * x_offset] = self.color(
                    face_point, face
                ).name[0]

        return "\n".join("".join(line) for line in reversed(result))

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
            if face is piece.facing:
                return piece.colors[0]
            return piece.colors[1]
        elif isinstance(piece, Corner):
            if face is piece.facing:
                return piece.colors[0]

            face_order = {
                (Face.RIGHT, Face.UP, Face.FRONT): (Face.RIGHT, Face.FRONT, Face.UP),
                (Face.RIGHT, Face.UP, Face.BACK): (Face.RIGHT, Face.UP, Face.BACK),
                (Face.RIGHT, Face.DOWN, Face.FRONT): (
                    Face.RIGHT,
                    Face.DOWN,
                    Face.FRONT,
                ),
                (Face.RIGHT, Face.DOWN, Face.BACK): (Face.RIGHT, Face.BACK, Face.DOWN),
                (Face.LEFT, Face.UP, Face.FRONT): (Face.LEFT, Face.UP, Face.FRONT),
                (Face.LEFT, Face.UP, Face.BACK): (Face.LEFT, Face.BACK, Face.UP),
                (Face.LEFT, Face.DOWN, Face.FRONT): (Face.LEFT, Face.FRONT, Face.DOWN),
                (Face.LEFT, Face.DOWN, Face.BACK): (Face.LEFT, Face.DOWN, Face.BACK),
            }[
                related_faces  # type: ignore
            ]

            if face_order[(face_order.index(piece.facing) + 1) % 3] is face:
                return piece.colors[1]

            return piece.colors[2]


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

    def timing(cube_size: int) -> None:
        import time

        print("Cube size:", cube_size)

        runs = 1000
        start = time.perf_counter()
        for i in range(runs):
            Cube(2)
        end = time.perf_counter()
        time_per_init = (end - start) / runs
        print(f"\ttpi={time_per_init:.2e} ips={1/time_per_init:.0f}")

        runs = 10000
        c = Cube(2)
        start = time.perf_counter()
        for i in range(runs):
            for face in Face:
                c.turn(face, -8)
        end = time.perf_counter()
        time_per_turn = (end - start) / runs / 6

        print(f"\ttpt={time_per_turn:.2e} tps={1/time_per_turn:.0f}")

    main()

    for cube_size in range(1, 100):
        # timing(cube_size)
        pass
