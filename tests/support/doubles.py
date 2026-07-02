"""TRIPWIRE — the double registry (Piece 1).

Every test double that SUBSTITUTES for a production object registers here via
``register_double(name, replaces)``. The returned instrument wraps the double so
that ANY invocation of it increments a per-test touch counter. The enforcement
hook in ``tests/conftest.py`` then fails any ``@pytest.mark.acceptance`` test
whose critical path touched a registered double — an acceptance test must
exercise the real path, and this makes "green over doubles" structurally
impossible instead of a human-review catch (the #73/#74 failure class).

What counts as a double (registry-worthy): fake providers, stubbed engines,
patched ``run_loop``/``_call_model``, fake bridges, in-memory stand-ins for the
real sqlite stores, patched routers/provider factories. What does NOT: data
builders, tmp_path plumbing, fixtures constructing REAL objects with test
config (those exercise the real thing and must never trip the wire).

Instrumentation strategy — in-place method wrapping, deliberately NOT a proxy
object: a wrapping proxy would change the double's identity and break
``isinstance`` checks against ABCs like ``ModelProvider`` that production code
performs. Instead:

* class double      → every function in the class's own ``__dict__`` (its
  substitute behavior; inherited REAL methods stay untouched) is wrapped to
  touch-then-delegate. Identity, subclassing and isinstance are preserved.
* function double   → returned wrapped (``functools.wraps``, so
  ``inspect.signature`` drift-pins keep working through ``__wrapped__``).
* instance double   → its class is instrumented (covers all instances).

Limitation (accepted, documented): a method attached to an already-registered
double AFTER registration is not wrapped. Register at definition time.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["register_double", "registry", "DoubleRecord"]


@dataclass
class DoubleRecord:
    """One registered double: identity + what it stands in for + touch count."""

    name: str
    replaces: str
    touched: int = 0


class DoubleRegistry:
    """Process-global registry of instrumented doubles.

    Registration is permanent for the process (module import time); ``touched``
    counters are zeroed before every test by the autouse fixture in
    ``tests/conftest.py``, so ``touched_records()`` reads "touched during the
    CURRENT test".
    """

    def __init__(self) -> None:
        self._records: dict[str, DoubleRecord] = {}

    def record(self, name: str, replaces: str) -> DoubleRecord:
        rec = self._records.get(name)
        if rec is None:
            rec = DoubleRecord(name=name, replaces=replaces)
            self._records[name] = rec
        return rec

    def reset(self) -> None:
        for rec in self._records.values():
            rec.touched = 0

    def touched_records(self) -> list[DoubleRecord]:
        return [r for r in self._records.values() if r.touched > 0]

    def touched_names(self) -> list[str]:
        return [r.name for r in self.touched_records()]

    def known_names(self) -> set[str]:
        return set(self._records)


registry = DoubleRegistry()


def _wrap_callable(fn: Callable[..., Any], rec: DoubleRecord) -> Callable[..., Any]:
    @functools.wraps(fn)
    def _touching(*args: Any, **kwargs: Any) -> Any:
        rec.touched += 1
        return fn(*args, **kwargs)

    _touching.__tripwire_record__ = rec  # type: ignore[attr-defined]
    return _touching


def _instrument_class(cls: type, rec: DoubleRecord) -> type:
    if getattr(cls, "__tripwire_record__", None) is rec:
        return cls  # idempotent (re-import / repeated fixture setup)
    for attr, val in list(vars(cls).items()):
        if attr.startswith("__") and attr != "__call__":
            continue  # dunders stay; __call__ IS substitute behavior
        if isinstance(val, staticmethod):
            setattr(cls, attr, staticmethod(_wrap_callable(val.__func__, rec)))
        elif isinstance(val, classmethod):
            setattr(cls, attr, classmethod(_wrap_callable(val.__func__, rec)))
        elif isinstance(val, property):
            fget = _wrap_callable(val.fget, rec) if val.fget else None
            setattr(cls, attr, property(fget, val.fset, val.fdel, val.__doc__))
        elif inspect.isfunction(val):
            setattr(cls, attr, _wrap_callable(val, rec))
    cls.__tripwire_record__ = rec  # type: ignore[attr-defined]
    return cls


def register_double(name: str, replaces: str) -> Callable[[Any], Any]:
    """Register + instrument a double. Usable as decorator or wrapper call.

    ``name``      unique registry key (shows up in the enforcement failure and
                  in ``acceptance(allow_doubles=[...])``).
    ``replaces``  the production object this double stands in for (dotted path
                  or plain description — it is printed in the failure message).

    Returns an *instrument* callable: apply it to the double (class, function,
    or instance) and use the result exactly as before — behavior is unchanged,
    only touch-counting is added.
    """
    rec = registry.record(name, replaces)

    def instrument(obj: Any) -> Any:
        if inspect.isclass(obj):
            return _instrument_class(obj, rec)
        if callable(obj):
            return _wrap_callable(obj, rec)
        _instrument_class(type(obj), rec)
        return obj

    return instrument
