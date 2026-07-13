---
name: skill-creator
description: Create, edit, improve, or audit Prometheus skills. Use when creating a new skill from scratch or when asked to improve, review, audit, tidy up, or clean up an existing skill. Also use when editing or restructuring a skill file. Triggers on phrases like "create a skill", "author a skill", "tidy up a skill", "improve this skill", "review the skill", "clean up the skill", "audit the skill".
---

# Skill Creator

A skill for creating and iteratively improving Prometheus skills.

## About Skills

Skills are modular, self-contained markdown files that extend Prometheus's capabilities by providing specialized knowledge, workflows, and tool guidance. They transform Prometheus from a general-purpose agent into a domain specialist equipped with procedural knowledge.

### What Skills Provide

1. Specialized workflows - Multi-step procedures for specific domains
2. Tool integrations - Instructions for working with specific file formats, APIs, or tools
3. Domain expertise - Project-specific knowledge, schemas, business logic
4. Bundled resources - Scripts, references, and assets for complex and repetitive tasks

### Core Principles

**Concise is key.** The context window is a shared resource. Only add context the agent does not already have. Challenge each piece: "Does the agent really need this explanation?" Prefer concise examples over verbose explanations.

**Set appropriate degrees of freedom.** Match specificity to task fragility:
- **High freedom** (text instructions): Multiple valid approaches, context-dependent decisions
- **Medium freedom** (pseudocode/scripts with parameters): Preferred pattern exists, some variation acceptable
- **Low freedom** (specific scripts, few parameters): Fragile operations, consistency critical

## Skill File Structure

Every skill is a single `.md` file in `~/.prometheus/skills/` with YAML frontmatter:

```markdown
---
name: skill-name
description: What this skill does and when to trigger it. Be comprehensive.
---

# Skill Name

Instructions and guidance for using the skill.
```

For complex skills that need bundled resources, use a directory:

```
skill-name/
  SKILL.md (required)
  scripts/          - Executable code (Python/Bash)
  references/       - Documentation loaded into context as needed
  assets/           - Files used in output (templates, etc.)
```

### Frontmatter

- `name`: The skill identifier (lowercase, hyphens, under 64 chars)
- `description`: Primary triggering mechanism. Include both what the skill does AND specific triggers/contexts. All "when to use" info goes here, not in the body.

### Body Guidelines

- Use imperative form in instructions
- Keep under 500 lines; split to reference files if approaching this limit
- Reference Prometheus tools by name: `bash`, `file_read`, `file_write`, `file_edit`, `grep`, `glob`, `tool_search`, `web_search`, `web_fetch`
- Reference Prometheus systems where relevant: LCM, wiki, SENTINEL

## Skill Creation Process

### Step 1: Capture Intent

Understand the user's intent. If the current conversation contains a workflow the user wants to capture, extract answers from conversation history first.

1. What should this skill enable the agent to do?
2. When should this skill trigger? (what user phrases/contexts)
3. What is the expected output format?
4. Are there test cases worth verifying against?

### Step 2: Interview and Research

Proactively ask about edge cases, input/output formats, example files, success criteria, and dependencies. Use `web_search` and `web_fetch` for research if useful.

### Step 3: Plan Reusable Contents

Analyze each concrete use case:
1. Consider how to execute from scratch
2. Identify what scripts, references, and assets would help when executing repeatedly

Example: A `pdf-editor` skill for "rotate this PDF" -> a `scripts/rotate_pdf.py` script avoids rewriting the same code each time.

### Step 4: Write the Skill

Remember the skill is for another instance of Prometheus to use. Include information that would be beneficial and non-obvious. Consider what procedural knowledge, domain-specific details, or reusable assets would help another agent instance.

**Writing patterns:**

- Define output formats with explicit templates
- Include examples (input/output pairs)
- Explain the **why** behind instructions rather than heavy-handed MUSTs
- Use theory of mind; make the skill general, not narrow to specific examples

**Skill naming:**
- Lowercase letters, digits, and hyphens only
- Short, verb-led phrases describing the action
- Namespace by tool when it improves clarity (e.g., `git-pr-review`, `telegram-notify`)

### Step 5: Test the Skill

After writing the skill draft, come up with 2-3 realistic test prompts. Share them with the user for validation.

For skills with objectively verifiable outputs (file transforms, data extraction, code generation), test cases are valuable. For subjective outputs (writing style, design), qualitative review is sufficient.

### Step 6: Iterate

After testing, improve based on feedback:

1. **Generalize from feedback** - The skill will be used across many different prompts. Avoid overfitting to test examples.
2. **Keep the skill lean** - Remove what is not pulling its weight.
3. **Explain the why** - Help the agent understand reasoning rather than blindly follow rules.
4. **Look for repeated work** - If every test case results in the agent writing similar helper scripts, bundle that script into the skill.

## Progressive Disclosure

Keep the body to essentials. Split content into separate files when approaching 500 lines. Reference those files clearly from the main skill with guidance on when to read them.

**Pattern 1: High-level guide with references**
```markdown
## Advanced features
- **Form filling**: See references/forms.md for complete guide
- **API reference**: See references/api.md for all methods
```

**Pattern 2: Domain-specific organization**
```
cloud-deploy/
  SKILL.md (workflow + selection)
  references/
    aws.md
    gcp.md
    azure.md
```
The agent reads only the relevant reference file.

## What NOT to Include

- README.md, CHANGELOG.md, INSTALLATION_GUIDE.md
- User-facing documentation or setup procedures
- Auxiliary context about the creation process
- Only include files needed for the agent to do the job

## Description Optimization

The description field is the primary triggering mechanism. To improve triggering accuracy:

1. Include both what the skill does AND specific contexts for when to use it
2. Be somewhat "pushy" in descriptions to avoid under-triggering
3. Cover different phrasings users might employ
4. Example: Instead of "Build dashboards", write "Build dashboards and data visualizations. Use whenever the user mentions dashboards, charts, metrics display, or data visualization, even if they do not explicitly ask for a 'dashboard.'"

## Prometheus-Specific Notes

- Skills live in `~/.prometheus/skills/` as `.md` files
- The agent loads skill metadata (name + description) to decide when to trigger
- The skill body is loaded only after triggering
- Available tools: `bash`, `file_read`, `file_write`, `file_edit`, `grep`, `glob`, `tool_search`, `web_search`, `web_fetch`
- For agent delegation, use the agent tool to dispatch subtasks
- For git operations, use `bash` with git commands directly
- For messaging, use the message tool or Telegram integration
