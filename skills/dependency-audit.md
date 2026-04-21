---
name: dependency-audit
description: "Use when auditing project dependencies for vulnerabilities, license compliance, supply chain risk, outdated packages, or dependency bloat. Supports JS, Python, Go, Rust, Ruby, Java, PHP, and .NET ecosystems."
version: 1.0.0
author: alirezarezvani/claude-skills
license: MIT
---
<!-- Provenance: alirezarezvani/claude-skills | engineering/dependency-auditor/SKILL.md | MIT -->

# Dependency Audit

Comprehensive dependency analysis, vulnerability scanning, license compliance, and supply chain security for multi-language projects.

## Vulnerability Scanning & CVE Matching

### What to Scan

Scan dependencies against vulnerability databases, match CVE patterns, analyze transitive dependency vulnerabilities, and provide CVSS scores.

### Supported Ecosystems

| Language | Manifest Files |
|----------|---------------|
| JavaScript/Node.js | package.json, package-lock.json, yarn.lock |
| Python | requirements.txt, pyproject.toml, Pipfile.lock, poetry.lock |
| Go | go.mod, go.sum |
| Rust | Cargo.toml, Cargo.lock |
| Ruby | Gemfile, Gemfile.lock |
| Java/Maven | pom.xml, gradle.lockfile |
| PHP | composer.json, composer.lock |
| C#/.NET | packages.config, project.assets.json |

### Ecosystem Audit Commands

```bash
# JavaScript
npm audit --json

# Python
pip audit --format json

# Go
govulncheck ./...

# Ruby
bundle audit check
```

### CVE Triage Workflow

1. **Collect** -- Run ecosystem audit tools, aggregate findings
2. **Deduplicate** -- Group by CVE ID across direct and transitive deps
3. **Prioritize** -- Critical + exploitable + reachable = fix immediately
4. **Remediate** -- Upgrade, patch, or mitigate with compensating controls
5. **Verify** -- Rerun audit to confirm fix, update lock files

## License Compliance

### License Classification

| Category | Licenses |
|----------|----------|
| Permissive | MIT, Apache 2.0, BSD (2/3-clause), ISC |
| Copyleft (Strong) | GPL (v2, v3), AGPL (v3) |
| Copyleft (Weak) | LGPL (v2.1, v3), MPL (v2.0) |
| Proprietary | Commercial, custom, restrictive |
| Dual Licensed | Multi-license scenarios |
| Unknown/Ambiguous | Missing or unclear licensing |

### Conflict Detection

- Identify incompatible license combinations
- Warn about GPL contamination in permissive projects
- Analyze license inheritance through dependency chains
- Provide compliance recommendations for distribution

## Outdated Dependency Detection

### Version Analysis

- Identify dependencies with available updates
- Categorize updates by severity (patch, minor, major)
- Detect pinned versions that may be outdated
- Analyze semantic versioning patterns
- Track last release dates and maintenance status

### Maintenance Status Red Flags

- No commits in 12+ months
- No releases in 12+ months
- Known end-of-life with no successor
- Security patches not being applied upstream
- Sole maintainer with no backup

## Dependency Bloat Analysis

### Unused Dependencies

- Identify dependencies that aren't actually imported/used
- Analyze import statements and usage patterns
- Detect redundant dependencies with overlapping functionality
- Identify oversized packages for simple use cases

### Redundancy Analysis

- Multiple packages providing similar functionality
- Version conflicts in transitive dependencies
- Bundle size impact analysis
- Opportunities for dependency consolidation

## Supply Chain Security

### Dependency Provenance

- Verify package signatures and checksums
- Identify suspicious or compromised packages
- Track package ownership changes and maintainer shifts
- Detect typosquatting and malicious packages

### Transitive Risk Analysis

- Map complete dependency trees
- Identify high-risk transitive dependencies
- Analyze dependency depth and complexity
- Provide supply chain risk scoring

## Upgrade Path Planning

### Risk Assessment

| Risk Level | Type | Action |
|-----------|------|--------|
| Low | Patch updates, security fixes | Apply immediately |
| Medium | Minor updates with new features | Test and apply within sprint |
| High | Major version updates, API changes | Plan migration, test thoroughly |
| Critical | Deps with known breaking changes | Dedicated migration effort |

### Upgrade Priority

1. Security patches -- highest priority
2. Bug fixes -- high priority
3. Feature updates -- medium priority
4. Major rewrites -- planned priority
5. Deprecated features -- immediate attention

## Lockfile Validation

- Ensure lockfiles are up-to-date with manifests
- Validate integrity hashes and version consistency
- Identify drift between environments (dev/staging/prod)
- Ensure deterministic, reproducible builds

## CI/CD Integration

```bash
# Security gate: fail on high/critical vulnerabilities
npm audit --audit-level=high
pip audit --fail-on-severity high

# License compliance check
# (use license-checker, license_finder, or pip-licenses)

# Pre-commit quick scan
npm audit --json | jq '.metadata.vulnerabilities.high + .metadata.vulnerabilities.critical'
```

## Scanning Frequency

| Scan Type | Frequency |
|-----------|-----------|
| Security scans | Daily or on every commit |
| License audits | Weekly or monthly |
| Upgrade planning | Monthly or quarterly |
| Full dependency audit | Quarterly |

## Best Practices

1. **Prioritize security** -- address high/critical CVEs immediately
2. **License first** -- ensure compliance before functionality
3. **Gradual updates** -- incremental dependency updates reduce risk
4. **Test thoroughly** -- comprehensive testing after every update
5. **Monitor continuously** -- automated alerting on new CVEs
6. **Review new deps** -- mandatory review before adding any dependency
7. **Document rationale** -- record why each dependency was chosen
8. **Maintain lockfiles** -- commit lockfiles, validate in CI
