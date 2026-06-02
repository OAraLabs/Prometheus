"""Standalone, cron-runnable Prometheus jobs.

Each module here exposes a ``main()`` and is runnable as
``python3 -m prometheus.jobs.<name>``. Jobs are deterministic pipelines that
reuse Prometheus components (providers, builtin tools) without an agent loop.
"""
