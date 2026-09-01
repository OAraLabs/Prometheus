"""One honest config read, and the four states it can end in.

WHY THIS MODULE EXISTS
----------------------
Eight subsystems used to read prometheus.yaml like this::

    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        section = data.get("<section>", {})
    except (OSError, Exception):
        section = {}

That handler converts "I could not read your configuration" into "you did not
configure anything." Those are different facts, and the code made them
indistinguishable — so a path that had never resolved to a real file on any
checkout survived two years and roughly six thousand tests. A wrong path that
raises is found on the first boot. A wrong path behind a bare except is found
by accident.

(``except (OSError, Exception)`` is also redundant: ``Exception`` subsumes
``OSError``. The belt-and-braces was the smell.)

Four modules under ``learning/`` were already remediated by the 2026-05
SILENT-FAILURE-AUDIT Tier-1 hotfix — narrowed catch, warning naming the path
and the substituted value. That remediation stopped at four of eight and left
no record of where it stopped, because the document it cited was never
committed. This module is the finish, and ``docs/audits/SILENT-FAILURE-AUDIT.md``
is its record.

THE FOUR STATES
---------------
``LOADED``      the file was read and parsed to a mapping.
``ABSENT``      no path was specified and no default file is present. Running
                on documented defaults is legitimate here — but it is now a
                RECORDED state rather than one inferred from silence.
``UNREADABLE``  a path WAS specified and could not be read: missing,
                permissions, a directory, a decode error. This is an error.
                Defaults are not a valid response to it.
``MALFORMED``   the file was read but did not parse to a mapping. Includes the
                state nobody had named: an EMPTY file makes ``safe_load``
                return ``None``, so the old ``data.get(...)`` raised
                ``AttributeError`` — swallowed by the same bare handler,
                indistinguishable from a missing file.

Plus ``PARTIAL``, which is per-key rather than per-file: the file loaded and a
key is absent. Legitimate, and recorded by :meth:`ConfigLoad.section` and
:meth:`ConfigLoad.value` rather than inferred.

WHAT IS LOUD
------------
UNREADABLE and MALFORMED are errors: ``log.error`` AND a row in the
``silent_failures`` ledger, which exists for exactly this and which no config
loader reached before this module. ABSENT and PARTIAL are legitimate, so they
log without a ledger row. Every one of them names the path attempted and the
value substituted — the standard the ``learning/`` four already met.

This module NEVER raises on a bad config: these are startup paths, and taking
the daemon down because an optional section is missing would be a worse
failure than the one being fixed. What it refuses to do is stay quiet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

LOADED = "loaded"
ABSENT = "absent"
UNREADABLE = "unreadable"
MALFORMED = "malformed"
PARTIAL = "partial"

#: States that mean a real failure — loud, and written to the ledger.
ERROR_STATES = frozenset({UNREADABLE, MALFORMED})

#: Everything a caller may see. Ordered for documentation, not comparison.
STATES = (LOADED, ABSENT, UNREADABLE, MALFORMED, PARTIAL)


class ConfigReadError(Exception):
    """Carries an UNREADABLE/MALFORMED read into the ledger.

    The ledger's ``record_silent_failure`` takes an exception because it stores
    a type, a message and a traceback. A malformed file produces no exception
    of its own, so one is synthesised here rather than passing ``None`` and
    losing the row.
    """


def _record(subsystem: str, operation: str, exc: BaseException,
            context: dict[str, Any]) -> None:
    """Best-effort ledger write. Never raises, never blocks a boot."""
    try:
        from prometheus.telemetry.tracker import get_telemetry_handle

        tel = get_telemetry_handle()
        if tel is not None and hasattr(tel, "record_silent_failure"):
            tel.record_silent_failure(subsystem, operation, exc, context)
    except Exception:  # noqa: BLE001 — telemetry must never break config loading
        log.debug("config load: ledger write failed", exc_info=True)


@dataclass
class ConfigLoad:
    """The outcome of one config read: the data AND how it went.

    ``data`` is always a mapping, so callers need no None-guard — but a caller
    that cares whether it is REAL checks ``state`` or ``ok``. That is the whole
    point: the substitution is still available, it just is no longer
    indistinguishable from a successful read.
    """

    data: dict[str, Any]
    state: str
    path: str | None
    detail: str
    subsystem: str = "config"
    _reported_keys: set[str] = field(default_factory=set, repr=False)

    @property
    def ok(self) -> bool:
        """True only when a real mapping was read from a real file."""
        return self.state == LOADED

    def section(self, name: str) -> dict[str, Any]:
        """A top-level section, recording PARTIAL when it is missing.

        Returns ``{}`` for a missing or non-mapping section — the same value
        the old code returned, now with a line saying it happened.
        """
        if not self.ok:
            # The file-level state was already reported; do not double-log a
            # missing section on a file that was never read.
            return {}
        value = self.data.get(name)
        if isinstance(value, dict):
            return value
        if value is None:
            self._report_partial(name, "{}")
        else:
            log.warning(
                "config %s: section %r in %s is %s, not a mapping — "
                "substituting {}",
                self.subsystem, name, self.path, type(value).__name__,
            )
        return {}

    def value(self, section: str, key: str, default: Any) -> Any:
        """A ``<section>.<key>`` scalar, recording PARTIAL when it is missing."""
        if not self.ok:
            return default
        sec = self.data.get(section)
        if not isinstance(sec, dict) or key not in sec:
            self._report_partial(f"{section}.{key}", default)
            return default
        return sec[key]

    def _report_partial(self, key: str, substituting: Any) -> None:
        if key in self._reported_keys:
            return
        self._reported_keys.add(key)
        log.info(
            "config %s: PARTIAL — %s is absent from %s; using the documented "
            "default %r",
            self.subsystem, key, self.path, substituting,
        )


def load_config_file(
    config_path: str | Path | None,
    *,
    subsystem: str,
    substituting: str,
    explicit: bool = True,
) -> ConfigLoad:
    """Read *config_path* and say, out loud, which of the four states resulted.

    Args:
        config_path: the file to read. ``None`` means the caller had nothing to
            read at all — reported as ABSENT.
        subsystem: short tag for the log line and the ledger row
            (``"security_gate"``, ``"token_budget"``, …).
        substituting: human description of what the caller falls back to, so
            every message names the consequence and not merely the cause.
        explicit: whether *config_path* was given BY THE CALLER. False means it
            came from a module-level default, which changes a missing file from
            an error into ABSENT: a caller that named a file wants that file,
            while a caller that named nothing is entitled to defaults — as long
            as it is on the record that it got them.
    """
    if config_path is None:
        detail = "no config path was specified"
        log.info(
            "config %s: ABSENT — %s; using %s",
            subsystem, detail, substituting,
        )
        return ConfigLoad({}, ABSENT, None, detail, subsystem)

    path = Path(config_path).expanduser()
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        if not explicit:
            # Nothing was asked for and nothing is there. Legitimate, but the
            # subsystem IS now running on defaults and that is worth a line.
            detail = f"no config at the default location {path}"
            log.warning(
                "config %s: ABSENT — %s; using %s. Every value this subsystem "
                "reads is a documented default, not your configuration.",
                subsystem, detail, substituting,
            )
            return ConfigLoad({}, ABSENT, str(path), detail, subsystem)
        detail = f"{path} does not exist"
        log.error(
            "config %s: UNREADABLE — %s; using %s. A path was specified and "
            "could not be read; defaults are not a valid answer to that.",
            subsystem, detail, substituting,
        )
        _record(subsystem, "load_config_file", exc,
                {"path": str(path), "state": UNREADABLE,
                 "substituting": substituting})
        return ConfigLoad({}, UNREADABLE, str(path), detail, subsystem)
    except (OSError, UnicodeDecodeError) as exc:
        # Permissions, a directory, a bad encoding, a vanished mount. Always an
        # error: something IS there and we could not read it.
        detail = f"{type(exc).__name__}: {exc}"
        log.error(
            "config %s: UNREADABLE — cannot read %s (%s); using %s",
            subsystem, path, detail, substituting,
        )
        _record(subsystem, "load_config_file", exc,
                {"path": str(path), "state": UNREADABLE,
                 "substituting": substituting})
        return ConfigLoad({}, UNREADABLE, str(path), detail, subsystem)
    except yaml.YAMLError as exc:
        detail = f"YAML parse failed: {exc}"
        log.error(
            "config %s: MALFORMED — %s is not valid YAML (%s); using %s",
            subsystem, path, exc, substituting,
        )
        _record(subsystem, "load_config_file", exc,
                {"path": str(path), "state": MALFORMED,
                 "substituting": substituting})
        return ConfigLoad({}, MALFORMED, str(path), detail, subsystem)

    if raw is None:
        # THE FOURTH STATE. An empty (or all-comment) file parses to None, and
        # the old `data.get(...)` raised AttributeError into the same bare
        # handler — so "your config file is empty" was indistinguishable from
        # "you have no config file". They call for different fixes.
        detail = f"{path} is empty — it parsed to None"
        exc = ConfigReadError(detail)
        log.error(
            "config %s: MALFORMED — %s; using %s. The file exists, so this is "
            "not a missing config: it has no content.",
            subsystem, detail, substituting,
        )
        _record(subsystem, "load_config_file", exc,
                {"path": str(path), "state": MALFORMED,
                 "substituting": substituting})
        return ConfigLoad({}, MALFORMED, str(path), detail, subsystem)

    if not isinstance(raw, dict):
        detail = f"{path} parsed to {type(raw).__name__}, not a mapping"
        exc = ConfigReadError(detail)
        log.error(
            "config %s: MALFORMED — %s; using %s",
            subsystem, detail, substituting,
        )
        _record(subsystem, "load_config_file", exc,
                {"path": str(path), "state": MALFORMED,
                 "substituting": substituting})
        return ConfigLoad({}, MALFORMED, str(path), detail, subsystem)

    log.debug("config %s: LOADED %s", subsystem, path)
    return ConfigLoad(raw, LOADED, str(path), "", subsystem)
