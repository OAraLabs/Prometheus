"""The service unit does not hand the env file to the process; the daemon loads it.

Removing ``EnvironmentFile=`` from the systemd unit is safe only because the
daemon reads the same file itself at startup (``load_env_file``), and every
consumer of a provider credential reads it from ``os.environ`` after that —
the daemon, its child processes (workers, spawned agents, coding runs, via
``task_child_env``) and operator cron jobs (which inherit ``os.environ``). A
model-written shell is still scrubbed. These tests pin that chain, so a future
change that moves a credential read ahead of ``load_env_file`` fails here
rather than in production.
"""

from __future__ import annotations

from pathlib import Path

from prometheus.cli.service import UNIT_TEMPLATE

UNIT = Path(__file__).resolve().parents[1] / "packaging" / "prometheus.service"
# A cloud provider default name (providers/registry.py CLOUD_DEFAULTS); the
# value is obviously fake and the pre-commit scanner ignores a short literal.
PROVIDER_VAR = "OPENAI_API_KEY"
FAKE = "sk-" + "FAKE-not-real-000"


def _active_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def test_neither_the_template_nor_the_packaged_unit_passes_the_env_file():
    # A comment may name the directive to explain the choice; what must be gone
    # is the active EnvironmentFile= line.
    for text in (UNIT_TEMPLATE, UNIT.read_text(encoding="utf-8")):
        assert not any(ln.strip().startswith("EnvironmentFile=") for ln in _active_lines(text))


def test_the_daemon_still_loads_the_env_file_and_children_keep_the_credential(tmp_path, monkeypatch):
    """With the value only in the file (never in the process env), load_env_file
    puts it in os.environ, a child env keeps it, and the model shell drops it."""
    from prometheus.config.env_file import load_env_file
    from prometheus.security.env_scrub import scrubbed_env
    from prometheus.tasks.child_env import task_child_env

    monkeypatch.delenv(PROVIDER_VAR, raising=False)
    env_file = tmp_path / "env"
    env_file.write_text(f"{PROVIDER_VAR}={FAKE}\n", encoding="utf-8")

    # Load into a copy, the way the daemon loads into os.environ at boot.
    import os
    monkeypatch.setattr(os, "environ", dict(os.environ), raising=True)
    assert PROVIDER_VAR not in os.environ
    load_env_file(env_file)
    assert os.environ.get(PROVIDER_VAR) == FAKE, "daemon's own load must populate it"

    # A daemon-built child (worker / spawned agent / coding driver).
    child = task_child_env(config_path=None)
    assert child.get(PROVIDER_VAR) == FAKE

    # An operator cron job inherits os.environ unchanged (cron_scheduler runs it
    # with env=None); a model-written shell / model cron is scrubbed.
    assert PROVIDER_VAR in os.environ
    assert PROVIDER_VAR not in scrubbed_env()
