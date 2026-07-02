"""Coverage-sentinel plumbing allowlist (TRIPWIRE checkpoint-1 condition).

Targets here are monkeypatch substitutions that are TEST PLUMBING, not
behavior doubles — redirecting paths/dirs, freezing environment lookups —
so the sentinel in tests/conftest.py must not warn about them. One line of
reason per entry. Behavior-substituting targets do NOT belong here; they
belong in the double registry (or in TRIPWIRE-2's backlog).

Format: "<owner>.<attr>" where owner is the patched object's module/class
name as the sentinel renders it.
"""

PLUMBING_TARGETS: set[str] = {
    # Path/dir redirection to tmp — the fixture builds REAL objects on temp roots.
    "prometheus.config.paths.get_config_dir",
    "prometheus.config.paths.get_data_dir",
    "pathlib.Path.home",
    "prometheus.morph.engine._daemon_lock_path",  # lock file under tmp, not behavior
    "prometheus.cron.service.get_cron_registry_path",  # registry file under tmp
}
