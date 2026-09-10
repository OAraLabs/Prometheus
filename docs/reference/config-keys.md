<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with:  uv run python scripts/gen_reference.py
     Pinned current by tests/test_generated_reference.py. -->

# Config key reference (generated)

Every key in `config/prometheus.yaml.default` — **396** of
them — with the value the template ships. This is what a fresh
install gets, not what the code falls back to when a key is absent:
those are pinned equal to each other by
`tests/test_config_defaults_equality.py`, and where they disagree
that file carries the debt register.

An empty value means the template ships no default and the code has
none either — the key is documented so its absence is visible.

| Key | Template value |
|---|---|
| `system` | *(section)* |
| `system.name` | `Prometheus` |
| `system.version` | `0.1.0` |
| `bootstrap` | *(section)* |
| `bootstrap.load_soul` | `True` |
| `bootstrap.load_agents` | `True` |
| `model` | *(section)* |
| `model.api_key_env` | `` |
| `model.api_key` | `` |
| `model.provider` | `llama_cpp` |
| `model.base_url` | `http://localhost:8080` |
| `model.model` | `` |
| `model.fallback` | *(section)* |
| `model.fallback.enabled` | `True` |
| `model.identity_probe_interval_seconds` | `300` |
| `model.grammar_enforcement` | `True` |
| `model.verify_thinking_suppression` | `warn` |
| `model.max_tool_iterations` | `500` |
| `model.max_tool_iterations_cloud` | `500` |
| `model.fallback_provider` | `ollama` |
| `model.fallback_url` | `http://localhost:11434` |
| `compaction` | *(section)* |
| `compaction.enabled` | `True` |
| `compaction.max_summary_tokens` | `512` |
| `compaction.protect_recent_turns` | `8` |
| `compaction.reserve_tokens` | `4096` |
| `compaction.threshold_pct` | `0.75` |
| `checkpoints` | *(section)* |
| `checkpoints.enabled` | `True` |
| `checkpoints.max_file_bytes` | `2000000` |
| `checkpoints.max_total_bytes` | `200000000` |
| `checkpoints.max_files` | `5000` |
| `checkpoints.keep_per_session` | `20` |
| `checkpoints.skip_dirs` | `['.git', '.hg', '.svn', 'node_modules', '.venv', 'venv', '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'dist', 'build', '.tox', '.prometheus']` |
| `sessions` | *(section)* |
| `sessions.rehydrate` | `False` |
| `context` | *(section)* |
| `context.effective_limit` | `24000` |
| `context.compression_trigger` | `0.75` |
| `context.tool_result_max` | `4000` |
| `context.tool_results_turn_budget` | `8000` |
| `context.reserved_output` | `2000` |
| `context.fresh_tail_count` | `32` |
| `context.microcompact_after_turns` | `3` |
| `context.microcompact_keep_chars` | `200` |
| `context.microcompact_keep_chars_no_lcm` | `500` |
| `context.cloud_default_limit` | `1000000` |
| `context.model_overrides` | `{}` *(open map)* |
| `context.microcompact_on_cloud` | `False` |
| `context.project_file_max_chars` | `12000` |
| `context.stack_project_files` | `True` |
| `context.project_files_max_total_chars` | `48000` |
| `tools` | *(section)* |
| `tools.deferred_loading` | *(section)* |
| `tools.deferred_loading.enabled` | `auto` |
| `tools.deferred_loading.always_loaded` | `['bash', 'task_create', 'read_file', 'write_file', 'edit_file', 'grep', 'glob', 'tool_search', 'web_search', 'web_fetch', 'memory']` |
| `tools.deferred_loading.mcp_always_deferred` | `False` |
| `tools.deferred_loading.search_mcp` | `True` |
| `adapter` | *(section)* |
| `adapter.adaptive_strictness` | `False` |
| `adapter.strictness_window` | `100` |
| `adapter.strictness_threshold` | `0.8` |
| `adapter.unwrap_dict_args` | *(empty — no code default)* |
| `security` | *(section)* |
| `security.permission_mode` | `default` |
| `security.workspace_root` | `~/.prometheus/workspace` |
| `security.denied_commands` | `['rm -rf /', 'rm -rf ~', 'DROP TABLE', 'mkfs']` |
| `security.denied_paths` | `['/etc', '/sys', '/boot', '/*/.ssh', '/*/.gnupg', '/*/.config/*/*env']` |
| `security.bash_confinement` | `off` |
| `security.bash_write_confinement` | `auto` |
| `security.bash_write_allow` | `[]` |
| `security.audit` | *(section)* |
| `security.audit.enabled` | `True` |
| `security.audit.retention_days` | `30` |
| `security.exfiltration` | *(section)* |
| `security.exfiltration.enabled` | `True` |
| `security.approval_queue` | *(section)* |
| `security.approval_queue.enabled` | `False` |
| `security.approval_queue.timeout_seconds` | `1800` |
| `security.allowed_commands` | `[]` |
| `infrastructure` | *(section)* |
| `infrastructure.archive_enabled` | `True` |
| `infrastructure.telemetry_enabled` | `True` |
| `infrastructure.gpu_host` | `` |
| `infrastructure.mini_host` | `` |
| `infrastructure.mini_port` | `8005` |
| `gateway` | *(section)* |
| `gateway.telegram_enabled` | `False` |
| `gateway.telegram_token` | `` |
| `gateway.allowed_chat_ids` | `[]` |
| `gateway.proxy_url` | *(empty — no code default)* |
| `gateway.heartbeat_interval` | `30` |
| `gateway.cron_enabled` | `True` |
| `gateway.skill_event_notifications` | `quiet` |
| `gateway.telegram` | *(section)* |
| `gateway.telegram.probe_interval_seconds` | `60` |
| `gateway.media` | *(section)* |
| `gateway.media.max_file_size_mb` | `20` |
| `gateway.media.cache_dir` | `~/.prometheus/cache/media` |
| `gateway.media.allowed_image_types` | `['image/jpeg', 'image/png', 'image/gif', 'image/webp']` |
| `gateway.media.allowed_audio_types` | `['audio/ogg', 'audio/mpeg', 'audio/wav']` |
| `gateway.media.allowed_document_types` | `['application/pdf', 'text/plain', 'text/markdown', 'text/csv', 'text/html', 'text/javascript', 'text/x-python', 'text/x-shellscript', 'text/typescript', 'application/json', 'application/sql', 'application/toml', 'application/xml', 'application/x-yaml']` |
| `gateway.media.cache_max_mb` | `512` |
| `gateway.media.free_disk_floor_mb` | `1024` |
| `gateway.rate_limits` | *(section)* |
| `gateway.rate_limits.messages_per_minute` | `30` |
| `gateway.rate_limits.media_downloads_per_minute` | `10` |
| `gateway.slack` | *(section)* |
| `gateway.slack.enabled` | `False` |
| `gateway.slack.bot_token` | `` |
| `gateway.slack.app_token` | `` |
| `gateway.slack.allowed_channels` | `[]` |
| `gateway.discord` | *(section)* |
| `gateway.discord.enabled` | `False` |
| `gateway.discord.token` | `` |
| `gateway.discord.guild_ids` | `[]` |
| `gateway.discord.channel_ids` | `[]` |
| `gateway.discord.user_ids` | `[]` |
| `gateway.discord.skill_event_notifications` | `quiet` |
| `gateway.discord.long_reply_threshold` | `800` |
| `gateway.paperclip` | *(section)* |
| `gateway.paperclip.enabled` | `False` |
| `gateway.paperclip.api_url` | `http://127.0.0.1:3100` |
| `gateway.paperclip.api_key` | `` |
| `gateway.paperclip.timeout_seconds` | `30` |
| `gateway.paperclip.comment_max_chars` | `4000` |
| `gateway.system_prompt` | `# see docs; overridden in live config` |
| `gateway.briefing_chat_id` | `` |
| `gateway.voice` | *(section)* |
| `gateway.voice.engine` | `piper` |
| `gateway.voice.default_mode` | `auto` |
| `gateway.voice.max_chars` | `800` |
| `gateway.voice.model_path` | `` |
| `gateway.voice.opus_bitrate` | `32k` |
| `gateway.slack_app_token` | `` |
| `gateway.slack_bot_token` | `` |
| `gateway.slack_channels` | *(empty — no code default)* |
| `gateway.slack_enabled` | `False` |
| `whisper` | *(section)* |
| `whisper.enabled` | `False` |
| `whisper.model` | `base` |
| `whisper.device` | `auto` |
| `whisper.language` | `en` |
| `router` | *(section)* |
| `router.overrides` | *(section)* |
| `router.overrides.enabled` | `True` |
| `router.overrides.sticky` | `True` |
| `router.fallback` | `[]` |
| `router.auxiliary` | *(section)* |
| `router.auxiliary.compression` | *(empty — no code default)* |
| `router.auxiliary.summarization` | *(empty — no code default)* |
| `router.auxiliary.vision` | *(empty — no code default)* |
| `router.escalation` | *(section)* |
| `router.escalation.as_subagent` | `True` |
| `router.escalation.budget_usd` | `1.0` |
| `router.escalation.enabled` | `False` |
| `router.escalation.provider` | *(empty — no code default)* |
| `router.rules` | `[]` |
| `router.smart_routing` | *(section)* |
| `router.smart_routing.enabled` | `False` |
| `router.smart_routing.max_simple_chars` | `160` |
| `router.smart_routing.max_simple_words` | `28` |
| `router.smart_routing.simple_provider` | *(empty — no code default)* |
| `slash_commands` | *(section)* |
| `slash_commands.claude` | *(section)* |
| `slash_commands.claude.provider` | `anthropic` |
| `slash_commands.claude.api_key_env` | `ANTHROPIC_API_KEY` |
| `slash_commands.claude.model` | `claude-haiku-4-5` |
| `slash_commands.gpt` | *(section)* |
| `slash_commands.gpt.provider` | `openai` |
| `slash_commands.gpt.api_key_env` | `OPENAI_API_KEY` |
| `slash_commands.gpt.model` | `gpt-4o` |
| `slash_commands.gemini` | *(section)* |
| `slash_commands.gemini.provider` | `gemini` |
| `slash_commands.gemini.api_key_env` | `GEMINI_API_KEY` |
| `slash_commands.gemini.model` | `gemini-2.5-flash` |
| `slash_commands.xai` | *(section)* |
| `slash_commands.xai.provider` | `xai` |
| `slash_commands.xai.api_key_env` | `XAI_API_KEY` |
| `slash_commands.xai.model` | `grok-4.5` |
| `slash_commands.deepseek` | *(section)* |
| `slash_commands.deepseek.provider` | `deepseek` |
| `slash_commands.deepseek.api_key_env` | `DEEPSEEK_API_KEY` |
| `slash_commands.deepseek.model` | `deepseek-v4-flash` |
| `slash_commands.kimi` | *(section)* |
| `slash_commands.kimi.provider` | `kimi` |
| `slash_commands.kimi.api_key_env` | `MOONSHOT_API_KEY` |
| `slash_commands.kimi.model` | `kimi-k2.6` |
| `slash_commands.glm` | *(section)* |
| `slash_commands.glm.provider` | `glm` |
| `slash_commands.glm.api_key_env` | `ZAI_API_KEY` |
| `slash_commands.glm.model` | `glm-5.2` |
| `slash_commands.mimo` | *(section)* |
| `slash_commands.mimo.provider` | `mimo` |
| `slash_commands.mimo.api_key_env` | `MIMO_API_KEY` |
| `slash_commands.mimo.model` | `mimo-v2.5-pro` |
| `divergence` | *(section)* |
| `divergence.enabled` | `False` |
| `divergence.threshold` | `0.7` |
| `divergence.checkpoint_interval` | `5` |
| `divergence.halt_on_repetition` | `True` |
| `learning` | *(section)* |
| `learning.nudge_enabled` | `True` |
| `learning.nudge_interval` | `15` |
| `learning.auto_skill_creation` | `True` |
| `learning.skill_min_tool_calls` | `3` |
| `learning.skill_refinement_enabled` | `False` |
| `learning.skill_refiner_model` | `default` |
| `learning.curator_enabled` | `True` |
| `learning.curator_interval_seconds` | `604800` |
| `learning.curator_stale_after_days` | `30` |
| `learning.curator_archive_after_days` | `90` |
| `learning.curator_min_idle_seconds` | `0` |
| `learning.curator_max_prunings_per_run` | `10` |
| `learning.curator_telegram_summary` | `True` |
| `learning.gepa_enabled` | `False` |
| `learning.gepa_max_skills_per_cycle` | `3` |
| `learning.gepa_variants_per_skill` | `3` |
| `learning.gepa_min_traces_required` | `10` |
| `learning.gepa_judge_threshold` | `0.7` |
| `learning.gepa_min_idle_minutes` | `10` |
| `learning.gepa_max_frequency_hours` | `24` |
| `learning.gepa_model` | `default` |
| `learning.live_recorder` | *(section)* |
| `learning.live_recorder.enabled` | `True` |
| `learning.live_recorder.verify_steps` | `True` |
| `learning.video_ingest` | *(section)* |
| `learning.video_ingest.enabled` | `False` |
| `learning.video_ingest.vision_model` | *(section)* |
| `learning.video_ingest.vision_model.provider` | `llama_cpp` |
| `learning.video_ingest.vision_model.base_url` | `http://localhost:8080` |
| `learning.video_ingest.vision_model.model` | `gemma-4` |
| `trajectory_export` | *(section)* |
| `trajectory_export.enabled` | `False` |
| `trajectory_export.interval_seconds` | `86400` |
| `trajectory_export.nightly_limit` | `1000` |
| `trajectory_export.output_dir` | `~/.prometheus/trajectories/` |
| `trajectory_export.format` | `jsonl` |
| `training` | *(section)* |
| `training.capture_enabled` | `True` |
| `training.cloud_golden_capture` | `False` |
| `training.db_path` | `~/.prometheus/data/training.db` |
| `coding` | *(section)* |
| `coding.enabled` | `False` |
| `coding.sandbox_type` | `process` |
| `coding.sandbox_dir` | `~/.prometheus/coding/sandboxes/` |
| `coding.max_task_duration_minutes` | `120` |
| `coding.max_iterations` | `50` |
| `coding.network_isolation` | `False` |
| `coding.docker_image` | `python:3.12-slim` |
| `coding.docker_cleanup_max_age_hours` | `24` |
| `coding.docker_cleanup_enabled` | `True` |
| `symbiote` | *(section)* |
| `symbiote.enabled` | `False` |
| `symbiote.github_token` | *(empty — no code default)* |
| `symbiote.language_default` | `Python` |
| `symbiote.min_stars_default` | `10` |
| `symbiote.max_repo_size_mb` | `100` |
| `symbiote.file_budget_max` | `15` |
| `symbiote.file_budget_kb` | `50` |
| `symbiote.clone_timeout_seconds` | `60` |
| `symbiote.sandbox_dir` | `~/.prometheus/symbiote/sandbox/` |
| `symbiote.harvest_dir` | `~/.prometheus/symbiote/harvests/` |
| `symbiote.sessions_db` | `~/.prometheus/symbiote/sessions.db` |
| `symbiote.scout_model` | `default` |
| `symbiote.harvest_model` | `default` |
| `symbiote.morph` | *(section)* |
| `symbiote.morph.enabled` | `False` |
| `symbiote.morph.health_check_timeout_seconds` | `60` |
| `symbiote.morph.health_check_interval_seconds` | `5` |
| `symbiote.morph.consecutive_passes_required` | `3` |
| `symbiote.morph.auto_rollback` | `True` |
| `symbiote.morph.daemon_manager` | `auto` |
| `symbiote.morph.daemon_health_url` | *(empty — no code default)* |
| `symbiote.morph.candidate_dir` | `~/.prometheus/symbiote/candidate/` |
| `symbiote.morph.post_mortem_dir` | `~/.prometheus/symbiote/post_mortem/` |
| `symbiote.backup` | *(section)* |
| `symbiote.backup.enabled` | `True` |
| `symbiote.backup.vault_root` | `~/.prometheus/symbiote/backups` |
| `symbiote.backup.max_backups` | `10` |
| `symbiote.backup.include_identity` | `True` |
| `symbiote.backup.include_config` | `True` |
| `symbiote.backup.pre_graft_backup` | `False` |
| `symbiote.backup.exempt_from_retention` | `['symbiote_morph', 'manual', 'pre_restore']` |
| `sentinel` | *(section)* |
| `sentinel.enabled` | `False` |
| `sentinel.idle_threshold_minutes` | `15` |
| `sentinel.dream_interval_minutes` | `30` |
| `sentinel.synthesis_enabled` | `True` |
| `sentinel.auto_fix_wiki` | `True` |
| `sentinel.dream_budget_tokens` | `2000` |
| `sentinel.stale_threshold_days` | `90` |
| `sentinel.confidence_decay_rate` | `0.05` |
| `sentinel.digest_lookback_hours` | `24` |
| `sentinel.nudge_cooldown_minutes` | `60` |
| `memory` | *(section)* |
| `memory.recall` | *(section)* |
| `memory.recall.enabled` | `True` |
| `memory.recall.max_facts` | `6` |
| `memory.recall.max_chars` | `900` |
| `memory.recall.min_confidence` | `0.6` |
| `web` | *(section)* |
| `web.enabled` | `True` |
| `web.api_port` | `8005` |
| `web.ws_port` | `8010` |
| `web.api_token` | *(empty — no code default)* |
| `web.dashboard_port` | `3002` |
| `push` | *(section)* |
| `push.enabled` | `False` |
| `push.apns` | *(section)* |
| `push.apns.key_path` | *(empty — no code default)* |
| `push.apns.key_id` | *(empty — no code default)* |
| `push.apns.team_id` | *(empty — no code default)* |
| `push.apns.topic` | *(empty — no code default)* |
| `image_generation` | *(section)* |
| `image_generation.default_backend` | `auto` |
| `image_generation.comfyui` | *(section)* |
| `image_generation.comfyui.base_url` | `http://127.0.0.1:8188` |
| `image_generation.comfyui.default_model` | `flux1-schnell-fp8.safetensors` |
| `image_generation.comfyui.default_steps` | `4` |
| `image_generation.dashscope` | *(section)* |
| `image_generation.dashscope.api_key_env` | `DASHSCOPE_API_KEY` |
| `image_generation.dashscope.model` | `wan2.5-t2i-preview` |
| `image_generation.dashscope.base_url` | `https://dashscope-intl.aliyuncs.com/api/v1` |
| `video_generation` | *(section)* |
| `video_generation.kling` | *(section)* |
| `video_generation.kling.access_key_env` | `KLING_ACCESS_KEY` |
| `video_generation.kling.secret_key_env` | `KLING_SECRET_KEY` |
| `video_generation.kling.base_url` | `https://api-singapore.klingai.com` |
| `video_generation.kling.model_name` | `kling-v3` |
| `video_generation.kling.poll_budget_seconds` | `600` |
| `web_tools` | *(section)* |
| `web_tools.fetch_timeout_seconds` | `30` |
| `web_tools.fetch_max_chars` | `8000` |
| `web_tools.search_max_results` | `8` |
| `web_tools.download_dir` | `~/.prometheus/downloads` |
| `web_tools.download_max_mb` | `100` |
| `web_tools.youtube_transcript_language` | `en` |
| `printing_press` | *(section)* |
| `printing_press.enabled` | `False` |
| `printing_press.library_path` | *(empty — no code default)* |
| `printing_press.auto_suggest` | `True` |
| `printing_press.auto_update_library` | `False` |
| `lsp` | *(section)* |
| `lsp.enabled` | `False` |
| `lsp.auto_diagnostics` | `True` |
| `lsp.diagnostics_delay_ms` | `500` |
| `lsp.servers` | `{}` *(open map)* |
| `profiles` | *(section)* |
| `profiles.default` | `full` |
| `profiles.custom_dir` | `~/.prometheus/profiles` |
| `backends` | `{}` *(open map)* |
| `backend_probe` | *(section)* |
| `backend_probe.ttl_s` | `60` |
| `backend_probe.timeout_s` | `5.0` |
| `anatomy` | *(section)* |
| `anatomy.enabled` | `True` |
| `anatomy.scan_on_startup` | `True` |
| `anatomy.include_in_system_prompt` | `True` |
| `anatomy.ssh_user` | `` |
| `anatomy.ssh_key` | `` |
| `mcp_servers` | `{}` *(open map)* |
| `hooks` | `{}` *(open map)* |
| `evals` | *(section)* |
| `evals.results_dir` | `~/.prometheus/eval_results` |
| `evals.skip_network_tasks` | `True` |
| `evals.judge_base_url` | `http://localhost:11434` |
| `evals.judge_model` | `qwen2.5:7b-instruct` |
| `tracing` | *(section)* |
| `tracing.enabled` | `False` |
| `tracing.service_name` | `prometheus` |
| `tracing.phoenix_endpoint` | `http://127.0.0.1:6006` |
| `wiki` | *(section)* |
| `wiki.root` | `~/.prometheus/wiki` |
| `vault` | *(section)* |
| `vault.root` | `~/brain-vault` |
| `vault.format_check` | `warn` |
| `doctor` | *(section)* |
| `doctor.startup_check` | `True` |
| `doctor.registry_file` | `config/model_registry.yaml` |
| `tasks` | *(section)* |
| `tasks.enabled` | `True` |
| `tasks.default_timeout_seconds` | `3600` |
| `tasks.poll_initial_interval_seconds` | `5` |
| `tasks.poll_max_interval_seconds` | `120` |
| `tasks.reengage_turn_cap` | `3` |
| `heartbeat` | *(section)* |
| `heartbeat.maintenance_db` | `` |
| `documents` | *(section)* |
| `documents.root` | *(empty — no code default)* |
| `escalation` | *(section)* |
| `escalation.api_key_env` | *(empty — no code default)* |
| `escalation.max_per_session` | `3` |
| `escalation.max_tokens` | `4096` |
| `escalation.teacher_model` | *(empty — no code default)* |
| `escalation.teacher_provider` | `anthropic` |
