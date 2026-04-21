---
name: security-engineering
description: "Use when performing threat modeling, security architecture review, vulnerability assessment, secure code review, or incident response. Provides STRIDE analysis, DREAD scoring, defense-in-depth patterns, cryptography selection, and security tooling reference."
version: 1.0.0
author: alirezarezvani/claude-skills
license: MIT
---
<!-- Provenance: alirezarezvani/claude-skills | engineering-team/senior-security/SKILL.md | MIT -->

# Security Engineering

Security engineering toolkit for threat modeling, vulnerability analysis, secure architecture design, and incident response.

## Threat Modeling (STRIDE)

### Workflow

1. **Define scope and boundaries:** Identify assets, map trust boundaries, document data flows
2. **Create data flow diagram:** External entities (users, services), processes (app components), data stores (databases, caches), data flows (APIs, network)
3. **Apply STRIDE to each DFD element** (see matrix below)
4. **Score risks using DREAD:** Damage (1-10), Reproducibility (1-10), Exploitability (1-10), Affected users (1-10), Discoverability (1-10)
5. **Prioritize** threats by risk score
6. **Define mitigations** for each threat
7. **Document** in threat model report

### STRIDE Threat Categories

| Category | Security Property | Mitigation Focus |
|----------|-------------------|------------------|
| Spoofing | Authentication | MFA, certificates, strong auth |
| Tampering | Integrity | Signing, checksums, validation |
| Repudiation | Non-repudiation | Audit logs, digital signatures |
| Information Disclosure | Confidentiality | Encryption, access controls |
| Denial of Service | Availability | Rate limiting, redundancy |
| Elevation of Privilege | Authorization | RBAC, least privilege |

### STRIDE per Element Matrix

| DFD Element | S | T | R | I | D | E |
|-------------|---|---|---|---|---|---|
| External Entity | X | | X | | | |
| Process | X | X | X | X | X | X |
| Data Store | | X | X | X | X | |
| Data Flow | | X | | X | X | |

## Security Architecture (Defense-in-Depth)

### Design Workflow

1. **Define security requirements:** Compliance (GDPR, HIPAA, PCI-DSS), data classification, threat model inputs
2. **Apply defense-in-depth layers:**
   - Perimeter: WAF, DDoS protection, rate limiting
   - Network: Segmentation, IDS/IPS, mTLS
   - Host: Patching, EDR, hardening
   - Application: Input validation, authentication, secure coding
   - Data: Encryption at rest and in transit
3. **Implement Zero Trust:** Verify explicitly (every request), least privilege (JIT/JEA), assume breach (segment, monitor)
4. **Configure auth:** Identity provider, MFA requirements, RBAC/ABAC model
5. **Design encryption strategy:** Key management, algorithm selection, certificate lifecycle
6. **Plan security monitoring:** Log aggregation, SIEM integration, alerting rules

### Authentication Pattern Selection

| Use Case | Recommended Pattern |
|----------|---------------------|
| Web application | OAuth 2.0 + PKCE with OIDC |
| API authentication | JWT with short expiration + refresh tokens |
| Service-to-service | mTLS with certificate rotation |
| CLI/Automation | API keys with IP allowlisting |
| High security | FIDO2/WebAuthn hardware keys |

## Vulnerability Assessment

### Workflow

1. **Define scope:** In-scope systems, testing methodology (black/gray/white box), rules of engagement
2. **Gather info:** Technology stack, architecture docs, prior vulnerability reports
3. **Automated scanning:** SAST, DAST, dependency scanning, secret detection
4. **Manual testing:** Business logic flaws, authentication bypass, authorization issues, injection
5. **Classify findings:**
   - Critical: Immediate exploitation risk
   - High: Significant impact, easier to exploit
   - Medium: Moderate impact or difficulty
   - Low: Minor impact
6. **Remediation plan:** Prioritize by risk, assign owners, set deadlines
7. **Verify fixes** and document

### Severity Matrix

| Impact \ Exploitability | Easy | Moderate | Difficult |
|-------------------------|------|----------|-----------|
| Critical | Critical | Critical | High |
| High | Critical | High | Medium |
| Medium | High | Medium | Low |
| Low | Medium | Low | Low |

## Secure Code Review

### Workflow

1. **Scope:** Changed files, security-sensitive areas (auth, crypto, input handling), third-party integrations
2. **Automated analysis:** SAST tools (Semgrep, CodeQL, Bandit), secret scanning, dependency vulnerability check
3. **Review authentication:** Password handling, session management, token validation
4. **Review authorization:** Access control checks, RBAC, privilege boundaries
5. **Review data handling:** Input validation, output encoding, SQL query construction, file path handling
6. **Review cryptography:** Algorithm selection, key management, random number generation
7. **Document findings** with severity

### Security Code Review Checklist

| Category | Check | Risk |
|----------|-------|------|
| Input Validation | All user input validated and sanitized | Injection |
| Output Encoding | Context-appropriate encoding applied | XSS |
| Authentication | Passwords hashed with Argon2/bcrypt | Credential theft |
| Session | Secure cookie flags (HttpOnly, Secure, SameSite) | Session hijacking |
| Authorization | Server-side permission checks on all endpoints | Privilege escalation |
| SQL | Parameterized queries used exclusively | SQL injection |
| File Access | Path traversal sequences rejected | Path traversal |
| Secrets | No hardcoded credentials or keys | Information disclosure |
| Dependencies | Known vulnerable packages updated | Supply chain |
| Logging | Sensitive data not logged | Information disclosure |

### Secure vs Insecure Patterns

| Pattern | Issue | Secure Alternative |
|---------|-------|-------------------|
| SQL string formatting | SQL injection | Parameterized queries with placeholders |
| Shell command building | Command injection | subprocess with argument lists, no shell |
| Path concatenation | Path traversal | Validate and canonicalize paths |
| MD5/SHA1 for passwords | Weak hashing | Argon2id or bcrypt |
| Math.random for tokens | Predictable values | crypto.getRandomValues |

## Incident Response

### Workflow

1. **Identify and triage:** Validate incident, assess scope/severity, activate response team
2. **Contain:** Isolate affected systems, block malicious IPs/accounts, disable compromised credentials
3. **Eradicate:** Remove malware/backdoors, patch vulnerabilities, update configurations
4. **Recover:** Restore from clean backups, verify integrity, monitor for recurrence
5. **Post-mortem:** Timeline reconstruction, root cause analysis, lessons learned
6. **Implement improvements:** Update detection rules, enhance controls, update runbooks

### Severity Levels

| Level | Response Time | Escalation |
|-------|---------------|------------|
| P1 - Critical (active breach) | Immediate | CISO, Legal, Executive |
| P2 - High (confirmed, contained) | 1 hour | Security Lead, IT Director |
| P3 - Medium (potential, investigating) | 4 hours | Security Team |
| P4 - Low (suspicious, low impact) | 24 hours | On-call engineer |

## Cryptographic Algorithm Selection

| Use Case | Algorithm | Key Size |
|----------|-----------|----------|
| Symmetric encryption | AES-256-GCM | 256 bits |
| Password hashing | Argon2id | N/A (use defaults) |
| Message authentication | HMAC-SHA256 | 256 bits |
| Digital signatures | Ed25519 | 256 bits |
| Key exchange | X25519 | 256 bits |
| TLS | TLS 1.3 | N/A |

## Security Headers Checklist

| Header | Recommended Value |
|--------|-------------------|
| Content-Security-Policy | default-src self; script-src self |
| X-Frame-Options | DENY |
| X-Content-Type-Options | nosniff |
| Strict-Transport-Security | max-age=31536000; includeSubDomains |
| Referrer-Policy | strict-origin-when-cross-origin |
| Permissions-Policy | geolocation=(), microphone=(), camera=() |

## Security Tools Reference

| Category | Tools |
|----------|-------|
| SAST | Semgrep, CodeQL, Bandit (Python), ESLint security plugins |
| DAST | OWASP ZAP, Burp Suite, Nikto |
| Dependency Scanning | Snyk, Dependabot, npm audit, pip-audit |
| Secret Detection | GitLeaks, TruffleHog, detect-secrets |
| Container Security | Trivy, Clair, Anchore |
| Infrastructure | Checkov, tfsec, ScoutSuite |
| Network | Wireshark, Nmap, Masscan |
| Penetration | Metasploit, sqlmap, Burp Suite Pro |
