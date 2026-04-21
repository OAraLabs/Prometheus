# Prometheus

## Project Rules
- Python 3.11+, package managed with uv
- All imports use `from prometheus.` prefix
- Config lives at config/prometheus.yaml, loaded via prometheus.config
- Run tests: uv run pytest tests/ -v
- All donor code has provenance headers (Source, License, Modified)
- Do not modify files in reference/

## Key Paths
- Tools: src/prometheus/tools/builtin/
- Adapter: src/prometheus/adapter/
- Engine: src/prometheus/engine/agent_loop.py
- Providers: src/prometheus/providers/
- Memory/LCM: src/prometheus/memory/
- Gateway: src/prometheus/gateway/telegram.py
- Config: config/prometheus.yaml
- Skills: skills/

## Conventions
- New tools extend BaseTool in tools/base.py
- Security checks go through SecurityGate (permissions/)
- Tool results truncated by tool_result_max in config
- ADDITIVE ONLY: extend existing files, don't replace them
