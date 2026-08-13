"""Tests for EMBERFALL dungeon generation."""

from __future__ import annotations

import random

import pytest

from emberfall.dungeon import (
    HEIGHT,
    WIDTH,
    generate_dungeon,
    is_fully_connected,
)


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 12345, 99999])
def test_dimensions(seed: int) -> None:
    d = generate_dungeon(level=1, seed=seed)
    assert d.width == WIDTH == 40
    assert d.height == HEIGHT == 20
    assert len(d.tiles) == HEIGHT
    assert all(len(row) == WIDTH for row in d.tiles)


@pytest.mark.parametrize("seed", range(20))
def test_always_fully_connected(seed: int) -> None:
    d = generate_dungeon(level=1, seed=seed)
    assert is_fully_connected(d), f"dungeon seed={seed} is not fully connected"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_stairs_only_before_level_3(level: int) -> None:
    d = generate_dungeon(level=level, seed=7)
    if level < 3:
        assert d.stairs_pos is not None
        sx, sy = d.stairs_pos
        assert d.tiles[sy][sx] == ">"
    else:
        assert d.stairs_pos is None
        flat = "".join("".join(row) for row in d.tiles)
        assert ">" not in flat


def test_has_rooms_and_floor() -> None:
    d = generate_dungeon(level=1, seed=3)
    assert len(d.rooms) >= 2
    floors = sum(1 for row in d.tiles for t in row if t in (".", ">"))
    assert floors > 10


def test_reproducible_with_seed() -> None:
    a = generate_dungeon(level=2, seed=123)
    b = generate_dungeon(level=2, seed=123)
    assert a.tiles == b.tiles
    assert [(r.x, r.y, r.w, r.h) for r in a.rooms] == [
        (r.x, r.y, r.w, r.h) for r in b.rooms
    ]


def test_borders_are_walls() -> None:
    """Rooms are carved inset so outer border should remain walls."""
    d = generate_dungeon(level=1, seed=5)
    # Corners always walls
    assert d.tiles[0][0] == "#"
    assert d.tiles[0][WIDTH - 1] == "#"
    assert d.tiles[HEIGHT - 1][0] == "#"
    assert d.tiles[HEIGHT - 1][WIDTH - 1] == "#"
