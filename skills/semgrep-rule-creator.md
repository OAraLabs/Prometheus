---
name: semgrep-rule-creator
description: "Creates custom Semgrep rules for detecting security vulnerabilities, bug patterns, and code patterns. Use when writing Semgrep rules, building custom static analysis detections, or enforcing coding standards via Semgrep."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/semgrep-rule-creator/skills/semgrep-rule-creator/SKILL.md | CC-BY-SA-4.0 -->

# Semgrep Rule Creator

Create production-quality Semgrep rules with proper testing and validation.

## When to Use

- Writing Semgrep rules for specific bug patterns
- Writing rules to detect security vulnerabilities in your codebase
- Writing taint mode rules for data flow vulnerabilities
- Writing rules to enforce coding standards

## When NOT to Use

- Running existing Semgrep rulesets
- General static analysis without custom rules

## Rationalizations to Reject

- **"The pattern looks complete"** -- Still run `semgrep --test` to verify. Untested rules have hidden false positives/negatives.
- **"It matches the vulnerable case"** -- Matching vulnerabilities is half the job. Verify safe cases don't match (false positives break trust).
- **"Taint mode is overkill for this"** -- If data flows from user input to a dangerous sink, taint mode gives better precision than pattern matching.
- **"One test is enough"** -- Include edge cases: different coding styles, sanitized inputs, safe alternatives, boundary conditions.
- **"I'll optimize the patterns first"** -- Write correct patterns first, optimize after all tests pass.
- **"The AST dump is too complex"** -- The AST reveals exactly how Semgrep sees code. Skipping it leads to patterns that miss syntactic variations.

## Anti-Patterns

**Too broad** -- matches everything, useless for detection:
```yaml
# BAD: Matches any function call
pattern: $FUNC(...)

# GOOD: Specific dangerous function
pattern: eval(...)
```

**Missing safe cases in tests** -- leads to undetected false positives:
```python
# BAD: Only tests vulnerable case
# ruleid: my-rule
dangerous(user_input)

# GOOD: Include safe cases
# ruleid: my-rule
dangerous(user_input)

# ok: my-rule
dangerous(sanitize(user_input))

# ok: my-rule
dangerous("hardcoded_safe_value")
```

**Overly specific patterns** -- misses variations:
```yaml
# BAD: Only matches exact format
pattern: os.system("rm " + $VAR)

# GOOD: Matches all os.system calls with taint tracking
mode: taint
pattern-sources:
  - pattern: input(...)
pattern-sinks:
  - pattern: os.system(...)
```

## Strictness Rules

- **Test-first is mandatory**: Never write a rule without tests
- **100% test pass is required**: "Most tests pass" is not acceptable
- **Optimization comes last**: Only simplify patterns after all tests pass
- **Avoid generic patterns**: Rules must be specific
- **Prioritize taint mode**: For data flow vulnerabilities
- **One YAML file -- one Semgrep rule**: Don't combine multiple rules in a single file
- **No generic rules**: When targeting a specific language, avoid `languages: generic`

## Approach Selection

- **Taint mode** (prioritize): Data flow issues where untrusted input reaches dangerous sinks
- **Pattern matching**: Simple syntactic patterns without data flow requirements

**Why prioritize taint mode?** Pattern matching finds syntax but misses context. A pattern `eval($X)` matches both `eval(user_input)` (vulnerable) and `eval("safe_literal")` (safe). Taint mode tracks data flow, reducing false positives dramatically for injection vulnerabilities.

**Iterating between approaches:** If taint mode isn't working well, switch to pattern matching. If pattern matching produces too many false positives, try taint mode. The goal is a working rule.

## Output Structure

Exactly 2 files in a directory named after the rule-id:
```
<rule-id>/
+-- <rule-id>.yaml     # Semgrep rule
+-- <rule-id>.<ext>    # Test file with ruleid/ok annotations
```

## Quick Start

```yaml
rules:
  - id: insecure-eval
    languages: [python]
    severity: HIGH
    message: User input passed to eval() allows code execution
    mode: taint
    pattern-sources:
      - pattern: request.args.get(...)
    pattern-sinks:
      - pattern: eval(...)
```

Test file (`insecure-eval.py`):
```python
# ruleid: insecure-eval
eval(request.args.get('code'))

# ok: insecure-eval
eval("print('safe')")
```

Run tests: `semgrep --test --config <rule-id>.yaml <rule-id>.<ext>`

## Workflow Checklist

```
Semgrep Rule Progress:
- [ ] Step 1: Analyze the Problem
- [ ] Step 2: Write Tests First
- [ ] Step 3: Analyze AST structure
- [ ] Step 4: Write the rule
- [ ] Step 5: Iterate until all tests pass (semgrep --test)
- [ ] Step 6: Optimize the rule (remove redundancies, re-test)
- [ ] Step 7: Final Run
```

### Step 1: Analyze the Problem

1. Understand the exact bug pattern
2. Identify the target language
3. Determine approach: pattern matching vs taint mode

### Step 2: Write Tests First

Create test file with `# ruleid:` and `# ok:` annotations. Must include:
- Clear vulnerable cases (must match)
- Clear safe cases (must not match)
- Edge cases and variations
- Different coding styles
- Sanitized/validated input (must not match)
- Nested structures (if/loops/try-catch)

**CRITICAL:** The annotation must be on the line IMMEDIATELY BEFORE the code.

### Step 3: Analyze AST Structure

```bash
semgrep --dump-ast --lang <language> <rule-id>.<ext>
```

### Step 4: Write the Rule

Use pattern operators from the Quick Reference below.

### Step 5: Iterate Until Tests Pass

```bash
semgrep --test --config <rule-id>.yaml <rule-id>.<ext>
# For taint mode debugging:
semgrep --dataflow-traces --config <rule-id>.yaml <rule-id>.<ext>
```

### Step 6: Optimize

Remove redundancies, then re-test:
- Quote variants (Semgrep normalizes `"string"` and `'string'`)
- Ellipsis subsets (`func($X, ...)` covers `func($X)`, `func($X, a, b)`)
- Consolidate similar patterns with `metavariable-regex`

### Step 7: Final Run

Run `semgrep --config <rule-id>.yaml <rule-id>.<ext>` and verify message has no uninterpolated metavariables.

## Pattern Quick Reference

### Basic Matching
```yaml
pattern: foo(...)                  # Basic match
patterns:                          # Logical AND
  - pattern: $X
  - pattern-not: safe($X)
pattern-either:                    # Logical OR
  - pattern: foo(...)
  - pattern: bar(...)
pattern-regex: ^foo.*bar$          # Regex
```

### Metavariables
- `$VAR` -- Match single expression (MUST be uppercase)
- `$_` -- Anonymous metavariable
- `$...VAR` -- Match zero or more arguments
- `...` -- Match anything in between
- `<... [pattern] ...>` -- Deep expression match

### Scope Operators
```yaml
pattern-inside: |
  def $FUNC(...):
    ...
pattern-not-inside: |
  with $CTX:
    ...
```

### Metavariable Filters
```yaml
metavariable-regex:
  metavariable: $FUNC
  regex: (unsafe|dangerous).*
metavariable-pattern:
  metavariable: $ARG
  pattern: request.$X
metavariable-comparison:
  metavariable: $NUM
  comparison: $NUM > 1024
```

### Taint Mode
```yaml
rules:
  - id: taint-rule
    mode: taint
    languages: [python]
    severity: HIGH
    message: Tainted data reaches sink
    pattern-sources:
      - pattern: user_input()
    pattern-sinks:
      - pattern: eval(...)
    pattern-sanitizers:
      - pattern: sanitize(...)
```

### Taint Options
```yaml
pattern-sources:
  - pattern: source(...)
    exact: true              # Only exact match is source
    by-side-effect: true     # Taints by side effect
pattern-sinks:
  - pattern: sink(...)
    exact: false             # Subexpressions also sinks
```

### Test Annotations
```python
# ruleid: rule-id
vulnerable_code()    # This line MUST match

# ok: rule-id
safe_code()          # This line MUST NOT match
```

### Debugging Commands
```bash
semgrep --test --config <rule-id>.yaml <rule-id>.<ext>
semgrep --validate --config <rule-id>.yaml
semgrep --dataflow-traces --config <rule-id>.yaml <rule-id>.<ext>
semgrep --dump-ast --lang <language> <rule-id>.<ext>
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Too many matches | Add `pattern-not` exclusions |
| Missing matches | Add `pattern-either` variants |
| Wrong line matched | Adjust `focus-metavariable` |
| Taint not flowing | Check sanitizers aren't too broad; use `--dataflow-traces` |
| Taint false positive | Add sanitizer pattern |
| Pattern not matching | Check AST with `--dump-ast`; verify metavariable binding |

## Documentation Links

Before writing any rule, read the official Semgrep docs:
1. [Rule Syntax](https://semgrep.dev/docs/writing-rules/rule-syntax)
2. [Pattern Syntax](https://semgrep.dev/docs/writing-rules/pattern-syntax)
3. [Testing Rules](https://semgrep.dev/docs/writing-rules/testing-rules)
4. [Taint Analysis](https://semgrep.dev/docs/writing-rules/data-flow/taint-mode)
5. [Advanced Taint](https://semgrep.dev/docs/writing-rules/data-flow/taint-mode/advanced)
6. [Constant Propagation](https://semgrep.dev/docs/writing-rules/data-flow/constant-propagation)
