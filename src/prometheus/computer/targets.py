"""Named machines, and which driver reaches each one.

WHY A REGISTRY RATHER THAN A DRIVER
------------------------------------
A desktop driver is a process the daemon talks to. Whether that process runs
on this machine or another one is a property of the CONNECTION, not a second
feature — so "drive the local desktop" and "drive another machine" are one
capability with two entries in a table, and the rest of the system never
branches on which.

THE NAME IS THE CONSENT TERM, AND THAT CONSTRAINS WHAT A NAME MAY BE
---------------------------------------------------------------------
``Target.name`` is the first term of every computer-use grant
(``target:app:verb:delivery``). Consequences, each load-bearing:

* **Operator-declared, never derived.** A name comes from config. It is never
  a hostname, an address, or anything read off the connection. Grants are
  persisted and the extent is rendered into the approval prompt, the audit
  row and the config file — a connection identifier in any of those breaks
  the standing rule against real infrastructure identifiers in persisted
  content. It would also let a connection detail *re-point* a grant: move a
  machine and every grant for it either silently dies or, far worse, silently
  transfers to whatever now answers there.
* **Stable.** Renaming a target invalidates its grants, by construction —
  they no longer match. That is the correct behaviour and it is worth knowing
  before renaming one.
* **Bounded.** Validated on construction, so a name cannot contain the
  delimiter and forge a different extent.

AN UNDECLARED TARGET IS REFUSED, NOT DEFAULTED
-----------------------------------------------
``resolve()`` raises for a name that is not in the table. There is no "fall
back to local", because the failure that would cause is the one this whole
term exists to prevent: an action meant for one machine landing on another.
A typo must be an error, not a quiet redirection to the nearest desktop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from prometheus.computer.driver import Driver, DriverUnavailable

#: A target name is a short lowercase token. Constrained rather than free text
#: because it is a grant term: ``:`` would forge a different extent, and
#: whitespace or case variants would split one machine's consent across
#: several records the operator never meant to create.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")

#: How a target is reached. ``local`` is the only kind milestone 1 implements.
KIND_LOCAL = "local"
KIND_REMOTE = "remote"
TARGET_KINDS: frozenset[str] = frozenset({KIND_LOCAL, KIND_REMOTE})


class UnknownTarget(KeyError):
    """No such declared target. Never falls back to another machine."""


@dataclass(frozen=True)
class Target:
    """One named machine Prometheus may drive."""

    name: str
    kind: str = KIND_LOCAL
    #: Free-form connection settings for the kind. Deliberately opaque here:
    #: the registry's job is naming and lookup, and a transport that leaks its
    #: shape into this module would make every transport this module's
    #: business. Values never reach the extent, the prompt or the audit row.
    connection: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"invalid target name {self.name!r}: a target name is the "
                f"first term of every computer-use grant, so it must be a "
                f"short lowercase token matching {_NAME_RE.pattern}"
            )
        if self.kind not in TARGET_KINDS:
            raise ValueError(
                f"unknown target kind {self.kind!r} for {self.name!r} "
                f"(expected one of {sorted(TARGET_KINDS)})"
            )


class TargetRegistry:
    """Declared targets, and the driver bound to each."""

    def __init__(self) -> None:
        self._targets: dict[str, Target] = {}
        self._drivers: dict[str, Driver] = {}

    def declare(self, target: Target, driver: Driver | None = None) -> None:
        """Add a target. A driver may be bound now or later."""
        if target.name in self._targets:
            raise ValueError(f"target {target.name!r} is already declared")
        self._targets[target.name] = target
        if driver is not None:
            self._drivers[target.name] = driver

    def bind(self, name: str, driver: Driver) -> None:
        """Attach a driver to an already-declared target."""
        if name not in self._targets:
            raise UnknownTarget(name)
        self._drivers[name] = driver

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._targets))

    def get(self, name: str) -> Target:
        try:
            return self._targets[name]
        except KeyError:
            raise UnknownTarget(
                f"{name!r} is not a declared target (declared: "
                f"{', '.join(self.names()) or 'none'}). Targets are named in "
                f"config; an undeclared name is refused rather than resolved "
                f"to another machine."
            ) from None

    def resolve(self, name: str) -> Driver:
        """The driver for *name*, or raise. NEVER falls back to another target."""
        target = self.get(name)  # raises UnknownTarget
        driver = self._drivers.get(name)
        if driver is None:
            raise DriverUnavailable(
                f"target {target.name!r} is declared ({target.kind}) but no "
                f"driver is connected to it — refusing rather than acting on "
                f"a different machine"
            )
        return driver


def registry_from_config(config: dict[str, Any] | None) -> TargetRegistry:
    """Build a registry from the ``computer_use.targets`` config block.

    Shape::

        computer_use:
          targets:
            local:
              kind: local
              description: this machine's desktop

    Declares targets only — binding a driver to each is the caller's job,
    because constructing a driver may connect to something and this function
    must not. A config with no targets yields an EMPTY registry, not an
    implicit local one: an implicit target is a machine nobody named, and its
    grants would read ``:firefox:click:background``.
    """
    registry = TargetRegistry()
    block = (config or {}).get("targets") or {}
    if not isinstance(block, dict):
        raise ValueError("computer_use.targets must be a mapping of name -> target")
    for name, spec in block.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"target {name!r} must be a mapping")
        registry.declare(Target(
            name=str(name),
            kind=str(spec.get("kind", KIND_LOCAL)),
            connection=dict(spec.get("connection") or {}),
            description=str(spec.get("description", "")),
        ))
    return registry
