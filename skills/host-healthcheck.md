---
name: host-healthcheck
description: Host security hardening and system health assessment. Use when asked for security audits, firewall/SSH hardening, system exposure review, or general host health checks on machines running Prometheus (laptops, workstations, servers, Pis).
version: 1.0.0
tags: [Security, Hardening, System-Health, Audit, Infrastructure]
---

# Host Healthcheck and Hardening

Assess and harden hosts in the Prometheus infrastructure. Covers firewall, SSH, updates, disk encryption, and general system health.

## Core Rules

- Require explicit approval before any state-changing action.
- Do not modify remote access settings without confirming how the user connects.
- Prefer reversible, staged changes with a rollback plan.
- Never claim Prometheus changes the host firewall, SSH, or OS updates; it does not.
- Format every set of user choices as numbered options so the user can reply with a single digit.

## Workflow

### 1. Establish Context (read-only)

Determine from the environment before asking the user:

1. OS and version (Linux/macOS), container vs host
2. Privilege level (root/admin vs user)
3. Access path (local console, SSH, Tailscale)
4. Network exposure (public IP, tunnel, LAN-only)
5. Backup system and status
6. Disk encryption status (FileVault/LUKS/BitLocker)
7. OS automatic security updates status

Ask once for permission to run read-only checks, then:

```bash
# OS info
uname -a
cat /etc/os-release 2>/dev/null || sw_vers 2>/dev/null

# Listening ports
ss -ltnup 2>/dev/null || lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null

# Firewall status (Linux)
ufw status 2>/dev/null || firewall-cmd --state 2>/dev/null || nft list ruleset 2>/dev/null

# Firewall status (macOS)
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null

# Tailscale status
tailscale status 2>/dev/null

# Disk space
df -h

# Memory
free -h 2>/dev/null || vm_stat 2>/dev/null

# Uptime and load
uptime
```

### 2. Determine Risk Tolerance

After system context is known, offer profiles:

1. **Home/Workstation Balanced** (most common): firewall on with reasonable defaults, remote access restricted to LAN or Tailnet
2. **Server Hardened**: deny-by-default inbound firewall, minimal open ports, key-only SSH, no root login, automatic security updates
3. **Developer Convenience**: more local services allowed, explicit exposure warnings, still audited
4. **Custom**: user-defined constraints

### 3. Produce Remediation Plan

Include:

- Target profile
- Current posture summary
- Gaps vs target
- Step-by-step remediation with exact commands
- Access-preservation strategy and rollback
- Risks and potential lockout scenarios
- Least-privilege notes
- Credential hygiene notes

Always show the plan before any changes.

### 4. Offer Execution Options

1. Do it for me (guided, step-by-step approvals)
2. Show plan only
3. Fix only critical issues
4. Export commands for later

### 5. Execute with Confirmations

For each step:

- Show the exact command
- Explain impact and rollback
- Confirm access will remain available
- Stop on unexpected output and ask for guidance

### 6. Verify and Report

Re-check:

- Firewall status
- Listening ports
- Remote access still works
- Overall posture summary

## Required Confirmations (always)

Require explicit approval for:

- Firewall rule changes
- Opening/closing ports
- SSH configuration changes
- Installing/removing packages
- Enabling/disabling services
- User/group modifications
- Scheduling tasks
- Update policy changes

## Prometheus Infrastructure Context

Common hosts to check (from `~/.prometheus/` config):

- **Mini** (Mac Mini): local workstation, Tailscale-connected
- **4090 GPU host**: remote compute, SSH + Tailscale
- **Raspberry Pi / VPS**: headless servers

Check Tailscale connectivity between hosts:

```bash
tailscale status
tailscale ping <hostname>
```

Check Prometheus-specific services:

```bash
# llama.cpp / Ollama endpoints
curl -s http://localhost:8080/health 2>/dev/null
curl -s http://localhost:11434/api/tags 2>/dev/null
```

## Periodic Checks

Recommended to run periodically:

```bash
# Quick system health
uname -a && uptime && df -h && free -h 2>/dev/null

# Security posture
ss -ltnup 2>/dev/null | grep LISTEN
ufw status 2>/dev/null

# Tailscale status
tailscale status
```

Store outputs and review for drift. Never log tokens or credential contents.
