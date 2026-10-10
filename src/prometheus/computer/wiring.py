"""The ONE place the daemon builds the door (computer-use v1.1 PR 5).

OFF BY DEFAULT, AND OFF MEANS NOTHING
-------------------------------------
``computer_use.enabled`` ships ``false`` and only a literal ``true`` turns it
on (``integration.computer_use_enabled``). While it is off this returns a
disabled integration and NO channel and NO runner: the driver is never
constructed, the Cua runtime is never started, every door route answers 404,
``/computer`` says it is off, and every surface keeps exactly the approval
queue it had before.

NOTHING IS REGISTERED — ON OR OFF
---------------------------------
The door is how a PERSON starts a desktop task. It adds no tool: a model
cannot call ``computer_task`` or any ``computer_*`` verb, and
``computer.registered`` stays 0 (``tests/test_computer_registration_pin.py``
builds the boot registry through this function with the switch off and on).
Registering ``computer_task`` for a model is L1, a separate ruling. The boot
registry is passed in only so the boot log can say so, by count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from prometheus.computer.integration import ComputerIntegration

logger = logging.getLogger(__name__)


@dataclass
class ComputerWiring:
    integration: ComputerIntegration
    channel: Any = None
    runner: Any = None
    config_errors: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.runner is not None

    def approvals_for(self, primary: Any) -> Any:
        """The answer surface every gateway and the web get: the gate-wide
        queue unchanged while computer use is off, both queues when on."""
        if self.channel is None:
            return primary
        from prometheus.permissions.approval_queue import ApprovalQueues

        if isinstance(primary, ApprovalQueues) and self.channel in primary.queues:
            return primary
        return ApprovalQueues(primary, self.channel)

    def attach_telegram(self, telegram: Any) -> None:
        """Desktop prompts reach the chat that started the task through the
        same adapter the gate-wide queue uses."""
        if self.channel is not None and telegram is not None:
            self.channel._telegram = telegram
            telegram._computer_runner = self.runner

    def attach_signal_bus(self, bus: Any) -> None:
        if self.channel is not None:
            self.channel.signal_bus = bus

    def close(self) -> None:
        """Daemon shutdown: stop every task, then the runtime."""
        if self.runner is not None:
            for task in self.runner.running():
                self.runner.stop(task.task_id)
        self.integration.close()


def _aliases(raw: Any, errors: list[str]) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        errors.append("computer_use.apps.aliases must be a mapping of "
                      "word -> list of app names")
        return {}
    out: dict[str, list[str]] = {}
    for word, names in raw.items():
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, (list, tuple)) or not all(
                isinstance(n, str) and n.strip() for n in names):
            errors.append(f"computer_use.apps.aliases.{word} must be a list "
                          f"of app names")
            continue
        out[str(word).strip().lower()] = [n.strip() for n in names]
    return out


def wire_computer_use(
    config: Mapping[str, Any] | None,
    *,
    gate: Any,
    registry: Any = None,
    device_store: Any = None,
    telegram_adapter: Any = None,
    telegram_user_ids: Iterable[int | str] = (),
    integration: ComputerIntegration | None = None,
    mcp_servers: Mapping[str, Any] | None = None,
    adapter_factory: Callable[[str], Any] | None = None,
) -> ComputerWiring:
    """Build the integration (unless given), and — only when enabled — the
    computer approval channel and the task runner. Never raises on config."""
    if integration is None:
        kwargs: dict[str, Any] = {"mcp_servers": mcp_servers}
        if adapter_factory is not None:
            kwargs["adapter_factory"] = adapter_factory
        integration = ComputerIntegration.from_config(config, **kwargs)
    wiring = ComputerWiring(integration=integration)
    if not integration.enabled:
        return wiring

    from prometheus.computer.approvals import ComputerApprovalChannel
    from prometheus.computer.door import PersonCheck
    from prometheus.computer.task import ComputerTaskRunner, TaskLimits

    block = (config or {}).get("computer_use") or {}
    errors: list[str] = []
    limits = TaskLimits.from_config(block.get("task"), errors)
    apps = block.get("apps") or {}
    aliases = _aliases(apps.get("aliases") if isinstance(apps, Mapping) else None,
                       errors)
    people = PersonCheck(device_store=device_store,
                         telegram_user_ids=telegram_user_ids)
    channel = ComputerApprovalChannel(security_gate=gate, people=people,
                                      telegram_adapter=telegram_adapter)
    runner = ComputerTaskRunner(integration=integration, gate=gate,
                                channel=channel, people=people, limits=limits,
                                aliases=aliases)
    wiring.channel = channel
    wiring.runner = runner
    wiring.config_errors = errors
    for err in errors:
        logger.warning("computer_use config: %s", err)
    if registry is not None:
        from prometheus.computer.status import _registered_count

        logger.info("Computer use: the door is open to people — "
                    "%s computer tool(s) registered for models",
                    _registered_count(registry))
    return wiring
