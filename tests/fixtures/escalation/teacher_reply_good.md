Here is the recovery for the failed turn.

```CORRECTIVE_REPLY
The deploy failed because the target host refused the connection, which means the network path is the problem rather than the script. Check that the VPN/tailnet link to the staging box is up, then rerun ./deploy.sh. If it still refuses, confirm the remote service is listening on the expected port before retrying.
```

```SKILL_DRAFT
---
name: diagnose-unreachable-deploy-target
description: Diagnose and recover when a deploy script cannot reach its target host.
---

# Diagnose Unreachable Deploy Target

## When to use
A deploy or remote script fails with connection-refused or unreachable errors.

## Steps
1. Confirm the network path (VPN/tailnet) is up before any retry.
2. Re-run the failing command and capture stderr.
3. Verify the remote service is listening on the expected port.

## Notes
- "Connection refused" is a host-side answer; "unreachable" is a routing problem — diagnose them differently.
```
