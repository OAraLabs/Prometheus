---
name: docker
description: Manage Docker containers, images, and compose stacks via the bash tool. Covers ps/logs/exec/restart/build/prune patterns, compose conventions, and safety rules to avoid destroying user data.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/docker.md | MIT -->

# Docker Workflow

Drive Docker and Docker Compose through the `bash` tool. Prometheus has no native `docker_*` tools — everything goes through `bash`. This skill teaches the *patterns* and *guardrails* so the agent acts predictably across the user's stacks.

## When to Use

✅ **USE this skill when:**

- The user mentions a container, compose stack, image, or volume
- Investigating why a service is down, slow, or misbehaving
- Bringing a stack up/down, rebuilding after a code change
- Tailing logs or exec-ing into a running container
- Cleaning up dangling images/volumes

## When NOT to Use

❌ **DON'T use this skill when:**

- The work is just editing files inside a project (use `file_edit`/`bash`)
- The host doesn't have docker installed (`command -v docker` first)
- The user is asking conceptual Docker questions, not operating their stacks

## Project Locations on This Box

Known compose/Dockerfile locations (as of skill authoring):

| Project | Path | Notes |
|---|---|---|
| openclaw | `~/openclaw/` | `Dockerfile`, `Dockerfile.sandbox`, `Dockerfile.sandbox-browser`, `docker-compose.yml` |
| ai-home-lab | `~/ai-home-lab/` | `docker-compose.yml` |
| n8n | `~/projects/n8n/` | `docker-compose.yml` |
| oara-voice | `~/projects/oara-voice/` | `docker-compose.yaml` + `docker-compose.cpu.yaml` (GPU/CPU variants) |

When the user says "the n8n stack" or "openclaw container", resolve to these paths. If unsure which project, ask before acting.

## Safe-by-Default Commands (Read-Only)

These never modify state. Run freely.

```bash
docker ps                          # running containers
docker ps -a                       # all, including stopped
docker images                      # all images
docker volume ls                   # volumes
docker network ls                  # networks
docker logs <name> --tail 200      # recent logs
docker logs <name> -f --tail 50    # follow (run in background, terminate when done)
docker inspect <name>              # full container detail
docker stats --no-stream           # one-shot resource snapshot
docker compose -f <path> ps        # compose stack status
docker compose -f <path> config    # rendered compose config (debug merges/overrides)
```

## State-Changing Commands (Confirm Intent First)

Reversible, but disrupt running services. Acceptable when the user has asked for them; otherwise confirm.

```bash
docker compose -f <path> up -d              # start stack detached
docker compose -f <path> down               # stop + remove containers (KEEPS volumes)
docker compose -f <path> restart <svc>      # restart one service
docker compose -f <path> build <svc>        # rebuild after Dockerfile/code change
docker compose -f <path> up -d --build <svc>  # rebuild + restart in one step
docker compose -f <path> pull               # pull newer images
docker restart <name>                       # restart single container
docker exec -it <name> <cmd>                # shell into running container
```

## Destructive Commands (NEVER Without Explicit User OK)

These delete data. Require an explicit user instruction naming the action.

```bash
docker compose down -v                  # ⚠️  removes VOLUMES (deletes data)
docker volume rm <vol>                  # deletes a volume
docker volume prune                     # deletes all unused volumes
docker system prune -a                  # deletes all unused images/containers/networks
docker system prune -a --volumes        # ⚠️  also deletes volumes
docker rm -f <name>                     # force-kill running container
docker rmi -f <image>                   # force-remove image
```

If the user says "clean up" or "free space", **default to the conservative path**:
```bash
docker image prune          # only dangling images
docker container prune      # only stopped containers
```
Report sizes before suggesting `system prune -a` or anything touching volumes.

## Common Patterns

### "Is X up?"
```bash
docker ps --filter "name=X" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### "Why is X failing?"
1. `docker ps -a --filter "name=X"` — is it stopped/restarting?
2. `docker logs <X> --tail 200` — recent errors
3. `docker inspect <X> --format '{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}'`
4. `docker events --since 10m --filter "container=X"` — restart loop signals

### "Restart after pulling code changes"
For compose stacks:
```bash
cd <project>
git pull
docker compose up -d --build <svc>     # rebuild only changed service
```

### "Free up disk"
Conservative cleanup, ordered by safety:
```bash
docker container prune -f       # stopped containers
docker image prune -f           # dangling images
docker builder prune -f         # build cache
# Stop here unless user OK'd more
```

### Compose file variant selection (oara-voice pattern)
```bash
# CPU variant
docker compose -f docker-compose.cpu.yaml up -d
# GPU (default) variant
docker compose -f docker-compose.yaml up -d
```
Always confirm which variant when GPU/CPU options exist.

## Working Directory Matters

`docker compose` resolves relative paths and `.env` files relative to the compose file's directory. Always either:
- `cd` into the project dir first, OR
- Use `-f /absolute/path/to/compose.yaml` and `--project-directory /absolute/path/`

For one-shot inspection, `-f` is fine. For `up -d`, prefer `cd` so `.env` and bind mounts resolve correctly.

## Anti-Patterns

- ❌ `docker system prune -af --volumes` without explicit user request — this is the "delete everything" button
- ❌ `docker compose down -v` when user only said "restart" — `-v` wipes data
- ❌ Force-removing containers (`rm -f`) before checking logs — you lose the failure evidence
- ❌ Running `docker pull` against an `image: latest` tag mid-incident — may introduce a new version into a known-broken state
- ❌ `docker exec` writes to a container's writable layer — they vanish on `docker compose up --force-recreate`. Persistent changes belong in volumes or the image.

## When the Host Has No Docker

Check first; don't assume:
```bash
command -v docker >/dev/null && docker version --format '{{.Server.Version}}' || echo "docker not available"
```

If missing, tell the user and stop — installing docker is a host-level decision they should make.

## Provenance and Audit

Container actions can have user-visible side effects (downtime, data loss). For destructive operations, follow the agent's `permissions/checker.py` flow: state intent, await confirmation, then act. Read-only ops can run freely.
