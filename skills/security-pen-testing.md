---
name: security-pen-testing
description: "Use when performing security audits, penetration testing, vulnerability scanning, OWASP Top 10 checks, or offensive security assessments. Covers static analysis, dependency scanning, secret detection, API security testing, and pen test reporting."
version: 1.0.0
author: alirezarezvani/claude-skills
license: MIT
---
<!-- Provenance: alirezarezvani/claude-skills | engineering-team/security-pen-testing/SKILL.md | MIT -->

# Security Penetration Testing

Offensive security testing methodology for finding vulnerabilities before attackers do. This is NOT compliance checking or security policy writing -- this is systematic vulnerability discovery through authorized testing.

**All testing requires written authorization from the system owner. Unauthorized testing is illegal.**

## Distinction from Other Security Skills

| Skill | Focus | Approach |
|-------|-------|----------|
| **security-pen-testing** (this) | Finding vulnerabilities | Offensive -- simulate attacker techniques |
| security-engineering | Threat modeling, architecture | STRIDE, defense-in-depth, policy |
| ai-security-assessment | AI/ML system security | Prompt injection, model inversion, ATLAS |
| adversarial-review | Code review quality | Hostile personas for code changes |

## OWASP Top 10 Quick Reference

| # | Category | Key Tests |
|---|----------|-----------|
| A01 | Broken Access Control | IDOR, vertical escalation, CORS, JWT claim manipulation, forced browsing |
| A02 | Cryptographic Failures | TLS version, password hashing, hardcoded keys, weak PRNG |
| A03 | Injection | SQLi, NoSQLi, command injection, template injection, XSS |
| A04 | Insecure Design | Rate limiting, business logic abuse, multi-step flow bypass |
| A05 | Security Misconfiguration | Default credentials, debug mode, security headers, directory listing |
| A06 | Vulnerable Components | Dependency audit (npm/pip/go), EOL checks, known CVEs |
| A07 | Auth Failures | Brute force, session cookie flags, session invalidation, MFA bypass |
| A08 | Integrity Failures | Unsafe deserialization, SRI checks, CI/CD pipeline integrity |
| A09 | Logging Failures | Auth event logging, sensitive data in logs, alerting thresholds |
| A10 | SSRF | Internal IP access, cloud metadata endpoints, DNS rebinding |

## Static Analysis

**Recommended tools:** CodeQL (custom queries), Semgrep (rule-based with auto-fix), ESLint security plugins.

**Key patterns to detect:**
- SQL injection via string concatenation
- Hardcoded JWT secrets
- Unsafe YAML/pickle deserialization
- Missing security middleware (e.g., Express without Helmet)

## Dependency Vulnerability Scanning

**Ecosystem commands:**
```bash
npm audit
pip audit
govulncheck ./...
bundle audit check
```

**CVE Triage Workflow:**
1. **Collect** -- run ecosystem audit tools, aggregate findings
2. **Deduplicate** -- group by CVE ID across direct and transitive deps
3. **Prioritize** -- critical + exploitable + reachable = fix immediately
4. **Remediate** -- upgrade, patch, or mitigate with compensating controls
5. **Verify** -- rerun audit to confirm fix, update lock files

## Secret Scanning

**Tools:** TruffleHog (git history + filesystem), Gitleaks (regex-based with custom rules).

```bash
# Scan git history for verified secrets
trufflehog git file://. --only-verified --json

# Scan filesystem
trufflehog filesystem . --json
```

**Integration:** Pre-commit hooks (gitleaks, trufflehog), CI/CD gates. Configure `.gitleaks.toml` for custom rules (AWS keys, API keys, private key headers) and allowlists for test fixtures.

## API Security Testing

### Authentication Bypass
- **JWT manipulation:** Change `alg` to `none`, RS256-to-HS256 confusion, claim modification (`role: "admin"`, `exp: 9999999999`)
- **Session fixation:** Check if session ID changes after authentication

### Authorization Flaws
- **IDOR/BOLA:** Change resource IDs in every endpoint -- test read, update, delete across users
- **BFLA:** Regular user tries admin endpoints (expect 403)
- **Mass assignment:** Add privileged fields (`role`, `is_admin`) to update requests

### Rate Limiting & GraphQL
- **Rate limiting:** Rapid-fire requests to auth endpoints; expect 429 after threshold
- **GraphQL:** Test introspection (disable in prod), query depth attacks, batch mutations bypassing rate limits

## Web Vulnerability Testing

| Vulnerability | Key Tests |
|--------------|-----------|
| XSS | Reflected (script/img/svg payloads), Stored (persistent fields), DOM-based (innerHTML + location.hash) |
| CSRF | Replay without token (expect 403), cross-session token replay, SameSite cookie attribute |
| SQL Injection | Error-based (`' OR 1=1--`), union-based enumeration, time-based blind (`SLEEP(5)`), boolean-based blind |
| SSRF | Internal IPs, cloud metadata endpoints (AWS/GCP/Azure), IPv6/hex/decimal encoding bypasses |
| Path Traversal | `../../../etc/passwd`, URL encoding, double encoding bypasses |

## Infrastructure Security

**Key checks:**
- **Cloud storage:** S3 bucket public access (`aws s3 ls s3://bucket --no-sign-request`), bucket policies, ACLs
- **HTTP security headers:** HSTS, CSP (no `unsafe-inline`/`unsafe-eval`), X-Content-Type-Options, X-Frame-Options, Referrer-Policy
- **TLS configuration:** Reject TLS 1.0/1.1, RC4, 3DES, export-grade ciphers (use `nmap --script ssl-enum-ciphers` or `testssl.sh`)
- **Port scanning:** `nmap -sV target.com` -- flag dangerous open ports (FTP/21, Telnet/23, Redis/6379, MongoDB/27017)

## Pen Test Report Structure

1. **Executive Summary:** Business impact, overall risk level, top 3 findings
2. **Scope:** What was tested, exclusions, testing dates
3. **Methodology:** Tools used, testing approach (black/gray/white box)
4. **Findings Table:** Sorted by severity with CVSS scores
5. **Detailed Findings:** Each with description, evidence, impact, remediation
6. **Remediation Priority Matrix:** Effort vs. impact for each fix
7. **Appendix:** Raw tool output, full payload lists

### Finding Format

```json
{
  "title": "SQL Injection in Login Endpoint",
  "severity": "critical",
  "cvss_score": 9.8,
  "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
  "category": "A03:2021 - Injection",
  "description": "The /api/login endpoint is vulnerable to SQL injection via the email parameter.",
  "evidence": "Request: POST /api/login {\"email\": \"' OR 1=1--\", \"password\": \"x\"}\nResponse: 200 OK with admin session token",
  "impact": "Full database access, authentication bypass, potential RCE",
  "remediation": "Use parameterized queries. Replace string concatenation with prepared statements.",
  "references": ["https://cwe.mitre.org/data/definitions/89.html"]
}
```

## Responsible Disclosure

Standard timeline: report day 1, follow up day 7, status update day 30, public disclosure day 90.

**Principles:**
- Never exploit beyond proof of concept
- Encrypt all communications
- Do not access real user data
- Document everything with timestamps

## Assessment Workflows

### Quick Security Check (15 Minutes)

1. Run OWASP checklist (quick scope)
2. Scan dependencies for high/critical CVEs
3. Check for secrets in recent commits
4. Review HTTP security headers

**Decision:** Any critical/high findings = block the merge.

### Full Penetration Test (Multi-Day)

**Day 1 -- Reconnaissance:**
- Map attack surface: endpoints, auth flows, third-party integrations
- Run full OWASP checklist
- Run dependency audit across all manifests
- Run secret scan on full git history

**Day 2 -- Manual Testing:**
- Authentication and authorization (IDOR, BOLA, BFLA)
- Injection points (SQLi, XSS, SSRF, command injection)
- Business logic flaws
- API-specific vulnerabilities (GraphQL, rate limiting, mass assignment)

**Day 3 -- Infrastructure and Reporting:**
- Cloud storage permissions
- TLS configuration and security headers
- Port scan for unnecessary services
- Compile findings and generate report

### CI/CD Security Gate

Automated checks on every PR: secret scanning, dependency audit, SAST (Semgrep with `p/security-audit`, `p/owasp-top-ten`), security headers on staging.

**Gate Policy:** Block on critical/high. Warn on medium. Log low/info.

## Anti-Patterns

1. **Testing in production without authorization** -- use staging/test environments when possible
2. **Ignoring low-severity findings** -- a chain of lows can become a critical exploit path
3. **Skipping responsible disclosure** -- every vulnerability must be reported through proper channels
4. **Relying solely on automated tools** -- tools miss business logic flaws and chained exploits
5. **Testing without defined scope** -- scope creep leads to legal liability
6. **Reporting without remediation guidance** -- every finding needs actionable remediation steps
7. **Storing evidence insecurely** -- pen test evidence is sensitive; encrypt and restrict access
8. **One-time testing** -- security testing must be continuous; integrate into CI/CD
