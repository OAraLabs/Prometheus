"""Tool-Calling Gym — deterministic experiment harness for open-model tool calling.

NOT an autonomous evolution loop. The gym runs FROZEN task sets against the
live model through the REAL pipeline (adapter, validator, hooks, telemetry),
one manifest = one variable changed vs baseline, N runs per task, results in
a dedicated store, and an honest report that refuses to declare winners on
thin samples. Humans decide the next experiment.

Layout:
  gym/tasksets/      — versioned YAML task sets (FROZEN per series)
  gym/experiments/   — experiment manifests (exactly one variable each)
  gym/results/       — committed reports per (series, experiment)
  ~/.prometheus/data/gym.db — gym_runs table (run-level records)

Entry point: ``uv run python scripts/gym_run.py --manifest gym/experiments/<m>.yaml``
"""

from prometheus.gym.manifest import ExperimentManifest, load_manifest
from prometheus.gym.tasks import TaskSpec, load_taskset

__all__ = [
    "ExperimentManifest",
    "TaskSpec",
    "load_manifest",
    "load_taskset",
]
