"""Tests for EMBERFALL save/load round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emberfall.dungeon import generate_dungeon
from emberfall.entities import Monster, Player, make_potion
from emberfall.main import new_game
from emberfall.save import (
    game_state_from_dict,
    game_state_to_dict,
    load_game,
    save_game,
    states_equal,
)


def test_save_load_roundtrip_identical(tmp_path: Path) -> None:
    state = new_game(seed=42)
    # Mutate a bit so it's not just defaults
    state["player"].hp = 17
    state["player"].inventory.append(make_potion(1, 2))
    state["turn"] = 7
    state["messages"].append("test message")
    if state["monsters"]:
        state["monsters"][0].hp = 3

    path = tmp_path / "save.json"
    save_game(state, path)
    loaded = load_game(path)

    assert states_equal(state, loaded)
    assert loaded["player"].hp == 17
    assert loaded["turn"] == 7
    assert loaded["player"].inventory[0].kind == "potion"
    assert len(loaded["monsters"]) == len(state["monsters"])
    assert loaded["dungeon"].tiles == state["dungeon"].tiles
    assert loaded["dungeon"].level == state["dungeon"].level


def test_roundtrip_dict_equality(tmp_path: Path) -> None:
    state = new_game(seed=7)
    path = tmp_path / "s.json"
    save_game(state, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    restored = game_state_from_dict(raw)
    assert game_state_to_dict(state) == game_state_to_dict(restored)


def test_save_creates_file(tmp_path: Path) -> None:
    state = new_game(seed=1)
    path = tmp_path / "nested" / "game.json"
    out = save_game(state, path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_player_position_persists(tmp_path: Path) -> None:
    state = new_game(seed=99)
    state["player"].x = 5
    state["player"].y = 6
    path = tmp_path / "p.json"
    save_game(state, path)
    loaded = load_game(path)
    assert loaded["player"].x == 5
    assert loaded["player"].y == 6


def test_boss_flag_persists(tmp_path: Path) -> None:
    state = new_game(seed=3)
    boss = Monster(
        name="Ashen Tyrant",
        x=2,
        y=2,
        hp=40,
        max_hp=40,
        attack=8,
        glyph="B",
        is_boss=True,
    )
    state["monsters"].append(boss)
    path = tmp_path / "b.json"
    save_game(state, path)
    loaded = load_game(path)
    bosses = [m for m in loaded["monsters"] if m.is_boss]
    assert len(bosses) == 1
    assert bosses[0].name == "Ashen Tyrant"
    assert bosses[0].hp == 40
