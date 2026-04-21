"""Default configuration values for Prometheus."""

from pathlib import Path

DEFAULTS_PATH = Path(__file__).parent.parent.parent.parent.parent / "config" / "prometheus.yaml"

DEFAULT_MODEL_PROVIDER = "llama_cpp"
DEFAULT_MODEL_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL_NAME = "qwen3.5-32b"

DEFAULT_CONTEXT_LIMIT = 24000
DEFAULT_COMPRESSION_TRIGGER = 0.75
DEFAULT_TOOL_RESULT_MAX = 4000
DEFAULT_RESERVED_OUTPUT = 2000
DEFAULT_FRESH_TAIL_COUNT = 32

DEFAULT_PERMISSION_MODE = "default"
