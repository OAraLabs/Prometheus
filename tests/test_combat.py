"""Tests for EMBERFALL combat and potions."""

from __future__ import annotations

import random

import pytest

from emberfall.combat import apply_death, resolve_attack, roll_damage, use_potion
from emberfall.entities import Monster, Player, make_potion


def test_damage_bounds() -> None:
    rng = random.Random(0)
    for attack in (1, 2, 4, 8, 10):
        for _ in range(50):
            dmg = roll_damage(attack, rng=rng)
            assert 1 <= dmg <= attack


def test_damage_zero_attack() -> None:
    assert roll_damage(0) == 0
    assert roll_damage(-3) == 0


def test_resolve_attack_reduces_hp() -> None:
    rng = random.Random(1)
    player = Player(hp=20, max_hp=20, attack=5, name="Hero")
    mon = Monster(name="Goblin", x=1, y=1, hp=10, max_hp=10, attack=2)
    result = resolve_attack(player, mon, rng=rng)
    assert result.damage >= 1
    assert mon.hp == 10 - result.damage or mon.hp == 0
    assert result.defender_hp == mon.hp
    assert result.killed == (mon.hp <= 0)


def test_death_handling() -> None:
    rng = random.Random(2)
    player = Player(hp=20, max_hp=20, attack=100, name="Hero")  # huge attack
    mon = Monster(name="Goblin", x=1, y=1, hp=3, max_hp=3, attack=1)
    result = resolve_attack(player, mon, rng=rng)
    assert result.killed is True
    assert mon.hp == 0
    assert mon.alive is False


def test_player_death() -> None:
    rng = random.Random(3)
    player = Player(hp=2, max_hp=20, attack=1, name="Hero")
    mon = Monster(name="Ogre", x=1, y=1, hp=20, max_hp=20, attack=50)
    result = resolve_attack(mon, player, rng=rng)
    assert result.killed is True
    assert player.hp == 0
    assert player.alive is False


def test_apply_death() -> None:
    mon = Monster(name="X", x=0, y=0, hp=5, max_hp=5, attack=1)
    apply_death(mon)
    assert mon.hp == 0
    assert mon.alive is False


def test_potion_healing() -> None:
    player = Player(hp=10, max_hp=20, attack=4)
    potion = make_potion()
    player.inventory.append(potion)
    msg = use_potion(player, potion)
    assert player.hp == 15
    assert "heals" in msg.lower() or "heal" in msg.lower()
    assert len(player.inventory) == 0


def test_potion_caps_at_max_hp() -> None:
    player = Player(hp=18, max_hp=20, attack=4)
    player.inventory.append(make_potion())
    use_potion(player)
    assert player.hp == 20


def test_potion_default_heal_amount_is_5() -> None:
    player = Player(hp=5, max_hp=20, attack=4)
    p = make_potion()
    assert p.heal_amount == 5
    player.inventory.append(p)
    use_potion(player)
    assert player.hp == 10


def test_no_potion_message() -> None:
    player = Player(hp=10, max_hp=20)
    msg = use_potion(player)
    assert "No potions" in msg
    assert player.hp == 10


def test_player_starts_with_20_hp() -> None:
    p = Player()
    assert p.hp == 20
    assert p.max_hp == 20
