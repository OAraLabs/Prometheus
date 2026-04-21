---
name: property-based-testing
description: "Guides property-based testing across multiple languages and smart contracts. Use when writing tests for serialization/parsing/normalization patterns, reviewing code with encode/decode pairs, or when PBT would provide stronger coverage than example-based tests."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/property-based-testing/skills/property-based-testing/SKILL.md | CC-BY-SA-4.0 -->

# Property-Based Testing Guide

Use this skill proactively during development when you encounter patterns where PBT provides stronger coverage than example-based tests.

## When to Invoke (Automatic Detection)

**Invoke this skill when you detect:**

- **Serialization pairs**: `encode`/`decode`, `serialize`/`deserialize`, `toJSON`/`fromJSON`, `pack`/`unpack`
- **Parsers**: URL parsing, config parsing, protocol parsing, string-to-structured-data
- **Normalization**: `normalize`, `sanitize`, `clean`, `canonicalize`, `format`
- **Validators**: `is_valid`, `validate`, `check_*` (especially with normalizers)
- **Data structures**: Custom collections with `add`/`remove`/`get` operations
- **Mathematical/algorithmic**: Pure functions, sorting, ordering, comparators
- **Smart contracts**: Solidity/Vyper contracts, token operations, state invariants, access control

**Priority by pattern:**

| Pattern | Property | Priority |
|---------|----------|----------|
| encode/decode pair | Roundtrip | HIGH |
| Pure function | Multiple | HIGH |
| Validator | Valid after normalize | MEDIUM |
| Sorting/ordering | Idempotence + ordering | MEDIUM |
| Normalization | Idempotence | MEDIUM |
| Builder/factory | Output invariants | LOW |
| Smart contract | State invariants | HIGH |

## When NOT to Use

- Simple CRUD operations without transformation logic
- One-off scripts or throwaway code
- Code with side effects that cannot be isolated (network calls, database writes)
- Tests where specific example cases are sufficient and edge cases are well-understood
- Integration or end-to-end testing (PBT is best for unit/component testing)
- UI/presentation logic
- Prototyping where requirements are fluid
- User explicitly requests example-based tests only

## Property Catalog (Quick Reference)

| Property | Formula | When to Use |
|----------|---------|-------------|
| **Roundtrip** | `decode(encode(x)) == x` | Serialization, conversion pairs |
| **Idempotence** | `f(f(x)) == f(x)` | Normalization, formatting, sorting |
| **Invariant** | Property holds before/after | Any transformation |
| **Commutativity** | `f(a, b) == f(b, a)` | Binary/set operations |
| **Associativity** | `f(f(a,b), c) == f(a, f(b,c))` | Combining operations |
| **Identity** | `f(x, identity) == x` | Operations with neutral element |
| **Inverse** | `f(g(x)) == x` | encrypt/decrypt, compress/decompress |
| **Oracle** | `new_impl(x) == reference(x)` | Optimization, refactoring |
| **Easy to Verify** | `is_sorted(sort(x))` | Complex algorithms |
| **No Exception** | No crash on valid input | Baseline property |

**Strength hierarchy** (weakest to strongest):
No Exception -> Type Preservation -> Invariant -> Idempotence -> Roundtrip

## How to Suggest PBT

When you detect a high-value pattern while writing tests, **offer PBT as an option**:

> "I notice `encode_message`/`decode_message` is a serialization pair. Property-based testing with a roundtrip property would provide stronger coverage than example tests. Want me to use that approach?"

**If codebase already uses a PBT library** (Hypothesis, fast-check, proptest, Echidna), be more direct:

> "This codebase uses Hypothesis. I'll write property-based tests for this serialization pair using a roundtrip property."

**If user declines**, write good example-based tests without further prompting.

## Rationalizations to Reject

Do not accept these shortcuts:

- **"Example tests are good enough"** -- If serialization/parsing/normalization is involved, PBT finds edge cases examples miss
- **"The function is simple"** -- Simple functions with complex input domains (strings, floats, nested structures) benefit most from PBT
- **"We don't have time"** -- PBT tests are often shorter than comprehensive example suites
- **"It's too hard to write generators"** -- Most PBT libraries have excellent built-in strategies; custom generators are rarely needed
- **"The test failed, so it's a bug"** -- Failures require validation; analyze whether it is a genuine bug, test error, or ambiguous specification
- **"No crash means it works"** -- "No exception" is the weakest property; always push for stronger guarantees

## Red Flags

- Recommending trivial getters/setters
- Missing paired operations (encode without decode)
- Ignoring type hints (well-typed = easier to test)
- Overwhelming user with candidates (limit to top 5-10)
- Being pushy after user declines

## Generating Property-Based Tests

### Process

1. **Analyze Target Function**: Read signature, types, docstrings; understand input types, output type, preconditions, invariants
2. **Design Input Strategies**: Build constraints INTO the strategy, not via `assume()`. Use realistic size limits.
3. **Identify Applicable Properties** from the Property Catalog above
4. **Generate Test Code**: Clear docstrings, appropriate settings, `@example` decorators for critical edge cases
5. **Include Edge Cases**: Always add explicit examples for empty, single-element, zero, negative, duplicates

### Settings Recommendations

```python
# Development (fast feedback)
@settings(max_examples=10)

# CI (thorough)
@settings(max_examples=200)

# Nightly/Release (exhaustive)
@settings(max_examples=1000, deadline=None)
```

### Example Test Patterns

**Roundtrip (Encode/Decode):**
```python
@given(valid_messages())
def test_roundtrip(msg):
    """Encoding then decoding returns original."""
    assert decode(encode(msg)) == msg
```

**Idempotence:**
```python
@given(st.text())
def test_normalize_idempotent(s):
    """Normalizing twice equals normalizing once."""
    assert normalize(normalize(s)) == normalize(s)
```

**Sorting Properties:**
```python
@given(st.lists(st.integers()))
@example([])
@example([1])
@example([1, 1, 1])
def test_sort(xs):
    result = sort(xs)
    assert len(result) == len(xs)        # Length preserved
    assert sorted(result) == sorted(xs)  # Elements preserved
    assert all(result[i] <= result[i+1] for i in range(len(result)-1))  # Ordered
    assert sort(result) == result        # Idempotent
```

**Validator + Normalizer:**
```python
@given(valid_inputs())
def test_normalized_is_valid(x):
    """Normalized inputs pass validation."""
    assert is_valid(normalize(x))
```

### Checklist Before Finishing

- [ ] Tests are not tautological (don't reimplement the function)
- [ ] At least one strong property (not just "no crash")
- [ ] Edge cases covered with `@example` decorators
- [ ] Strategy constraints are realistic, not over-filtered
- [ ] Settings appropriate for context (dev vs CI)
- [ ] Docstrings explain what each property verifies
- [ ] Tests actually run and pass (or fail for expected reasons)

## PBT Libraries by Language

| Language | Library | Import/Setup |
|----------|---------|--------------|
| Python | Hypothesis | `from hypothesis import given, strategies as st` |
| JavaScript/TypeScript | fast-check | `import fc from 'fast-check'` |
| Rust | proptest | `use proptest::prelude::*` |
| Go | rapid | `import "pgregory.net/rapid"` |
| Java | jqwik | `@Property` annotations, `import net.jqwik.api.*` |
| Scala | ScalaCheck | `import org.scalacheck._` |
| C# | FsCheck | `using FsCheck; using FsCheck.Xunit;` |
| Elixir | StreamData | `use ExUnitProperties` |
| Haskell | QuickCheck | `import Test.QuickCheck` |
| Clojure | test.check | `[clojure.test.check :as tc]` |
| Ruby | PropCheck | `require 'prop_check'` |
| Kotlin | Kotest | `io.kotest.property.*` |
| C++ | RapidCheck | `#include <rapidcheck.h>` |

### Smart Contract Testing (EVM/Solidity)

| Tool | Type | Description |
|------|------|-------------|
| Echidna | Fuzzer | Property-based fuzzer for EVM contracts |
| Medusa | Fuzzer | Next-gen fuzzer with parallel execution |

```solidity
// Echidna property example
function echidna_balance_invariant() public returns (bool) {
    return address(this).balance >= 0;
}
```

### Detecting Existing Usage

```bash
# Python
rg "from hypothesis import" --type py

# JavaScript/TypeScript
rg "from 'fast-check'" --type js --type ts

# Rust
rg "use proptest" --type rust

# Go
rg "pgregory.net/rapid" --type go

# Solidity (Echidna)
rg "echidna_" --glob "*.sol"
```
