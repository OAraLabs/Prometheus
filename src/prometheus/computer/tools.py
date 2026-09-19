"""The wrapped tools as ``BaseTool`` instances — what the dispatch path sees.

These exist so the gate is reached through the SAME machinery every other tool
is reached through: ``_execute_tool_call`` reads ``tool.input_model``, derives
the extent from its schema, and calls ``evaluate``. Nothing about computer use
gets its own private route to the gate, because a private route is a route
that can be forgotten.

``is_read_only`` is honest rather than convenient. ``observe`` and ``verify``
read; everything else mutates a desktop the operator is also using. It is NOT
derived from a driver's self-declaration — the MCP consent survey's ruling
(``readOnlyHint`` is not trusted) applies with more force here, since these
actions have no floor beneath them the way a file read does.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from prometheus.computer.actions import (
    ACTION_MODELS,
    ALLOWED_KEYS,
    ClickInput,
    InvokeMenuInput,
    ObserveInput,
    PressKeyInput,
    ScrollInput,
    TypeTextInput,
    VerifyInput,
)
from prometheus.computer.driver import Driver, DriverUnavailable, StaleSnapshot
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

#: Verbs that only read. Everything else changes the operator's desktop.
_READ_ONLY_VERBS: frozenset[str] = frozenset({"observe", "verify"})


class _ComputerTool(BaseTool):
    """One wrapped desktop action."""

    verb: str = ""

    def __init__(self, driver: Driver | None = None) -> None:
        self._driver = driver
        self.name = f"computer_{self.verb}"
        self.description = self.__doc__ or f"Desktop action: {self.verb}"

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return self.verb in _READ_ONLY_VERBS

    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext
    ) -> ToolResult:
        del context
        if self._driver is None:
            # Not "return an empty result". A tool whose driver is absent must
            # say so — a computer-use call that quietly no-ops is the exact
            # failure the precondition check exists to prevent, and it would
            # be indistinguishable from an action that landed.
            return ToolResult(
                output=(
                    f"{self.name}: no desktop driver is configured — refusing "
                    f"rather than reporting an action that did not happen"
                ),
                is_error=True,
            )
        args: dict[str, Any] = arguments.model_dump(exclude_none=True)
        if self.verb == "press_key":
            key = str(args.get("key", "")).lower()
            if key not in ALLOWED_KEYS:
                # Refused BEFORE the driver. The closed key set is what makes
                # "press keys in <app>" a describable extent; a tool that
                # forwarded arbitrary key names would quietly widen the grant
                # its own consent sentence promised.
                return ToolResult(
                    output=(
                        f"{self.name}: key {key!r} is not in the allowed set "
                        f"({', '.join(sorted(ALLOWED_KEYS))})"
                    ),
                    is_error=True,
                )
        try:
            result = self._driver.act(self.verb, args)
        except StaleSnapshot as exc:
            return ToolResult(output=f"{self.name}: {exc}", is_error=True)
        except DriverUnavailable as exc:
            return ToolResult(output=f"{self.name}: {exc}", is_error=True)
        except Exception as exc:  # pragma: no cover - driver-specific
            log.warning("%s failed", self.name, exc_info=True)
            return ToolResult(output=f"{self.name}: {exc}", is_error=True)
        return ToolResult(output=str(result))


class ComputerObserveTool(_ComputerTool):
    """Read a window's accessibility tree and return its elements."""

    verb = "observe"
    input_model = ObserveInput


class ComputerClickTool(_ComputerTool):
    """Click one element identified by a snapshot-bound token."""

    verb = "click"
    input_model = ClickInput


class ComputerScrollTool(_ComputerTool):
    """Scroll a window."""

    verb = "scroll"
    input_model = ScrollInput


class ComputerPressKeyTool(_ComputerTool):
    """Press one named key from a closed set."""

    verb = "press_key"
    input_model = PressKeyInput


class ComputerTypeTextTool(_ComputerTool):
    """Insert text into an element. Never remembered — see the payload rule."""

    verb = "type_text"
    input_model = TypeTextInput


class ComputerInvokeMenuTool(_ComputerTool):
    """Invoke a menu path."""

    verb = "invoke_menu"
    input_model = InvokeMenuInput


class ComputerVerifyTool(_ComputerTool):
    """Check an expectation against fresh window state."""

    verb = "verify"
    input_model = VerifyInput


#: Every wrapped tool class, keyed by verb. Derived from the same ACTION_MODELS
#: table the schemas come from, and checked against it at import time so a verb
#: cannot gain a model without gaining a tool (or the reverse).
TOOL_CLASSES: dict[str, type[_ComputerTool]] = {
    "observe": ComputerObserveTool,
    "click": ComputerClickTool,
    "scroll": ComputerScrollTool,
    "press_key": ComputerPressKeyTool,
    "type_text": ComputerTypeTextTool,
    "invoke_menu": ComputerInvokeMenuTool,
    "verify": ComputerVerifyTool,
}

assert set(TOOL_CLASSES) == set(ACTION_MODELS), (
    "computer verbs drifted: TOOL_CLASSES and ACTION_MODELS must agree "
    f"({sorted(set(TOOL_CLASSES) ^ set(ACTION_MODELS))})"
)


def build_computer_tools(driver: Driver | None = None) -> list[_ComputerTool]:
    """Every wrapped computer tool, bound to *driver*."""
    return [cls(driver) for cls in TOOL_CLASSES.values()]


def register_computer_tools(registry: Any, driver: Driver | None = None) -> int:
    """Register the wrapped tools with a ToolRegistry. Returns the count.

    ⚠ NOT called from ``create_tool_registry``. Registering these makes a chat
    model able to click and type directly, which is a blast-radius decision
    separate from making the gate able to rule on such a call — and only the
    second is milestone 1. The function exists so the tests can drive the REAL
    dispatch path; wiring it into the daemon is a later, deliberate change.
    """
    count = 0
    for tool in build_computer_tools(driver):
        registry.register(tool)
        count += 1
    return count
