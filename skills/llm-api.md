---
name: llm-api
description: "Build apps with LLM APIs (Anthropic, OpenAI, local models via llama.cpp/Ollama). Use when code imports AI SDKs (anthropic, openai, ollama, etc.), or the user asks to use any LLM API. DO NOT USE for general programming or ML/data-science tasks unrelated to LLM integration."
license: CC-BY-SA-4.0
---

# Building LLM-Powered Applications

This skill helps build LLM-powered applications using various providers. Choose the right provider and surface based on needs, detect the project language, then implement accordingly.

## Provider Detection

Determine which LLM provider the user is working with:

1. **Look at project files and imports** to infer the provider:
   - `anthropic` / `@anthropic-ai/sdk` -> Anthropic (Claude)
   - `openai` / `OpenAI` -> OpenAI (GPT)
   - `ollama` -> Ollama (local models)
   - `requests` to localhost:8080 or similar -> llama.cpp server
   - If unclear, ask the user

2. **For local infrastructure** (Prometheus context):
   - Ollama on Mini: `http://<host>:11434`
   - llama.cpp on 4090: `http://<host>:8080`
   - These can serve as local LLM backends

## Language Detection

Before reading code examples, determine which language the user is working in:

- `*.py`, `requirements.txt`, `pyproject.toml` -> **Python**
- `*.ts`, `*.tsx`, `package.json`, `tsconfig.json` -> **TypeScript**
- `*.js`, `*.jsx` (no `.ts` files) -> **JavaScript** (same SDK as TS)
- `*.go`, `go.mod` -> **Go**
- `*.rs`, `Cargo.toml` -> **Rust**

Use `glob` to find project files and `file_read` to examine them.

## Which Surface Should I Use?

> **Start simple.** Default to the simplest tier that meets your needs.

| Use Case | Tier | Recommended Surface |
| --- | --- | --- |
| Classification, summarization, extraction, Q&A | Single LLM call | Direct API call |
| Batch processing | Single LLM call | Batch endpoint or loop |
| Multi-step pipelines with code-controlled logic | Workflow | API + tool use |
| Custom agent with your own tools | Agent | API + tool use loop |
| AI agent with file/web/terminal access | Agent | Agent SDK or custom harness |

### Decision Tree

```
What does your application need?

1. Single LLM call (classification, summarization, extraction, Q&A)
   -> Direct API call - one request, one response

2. Does the LLM need to use tools autonomously?
   -> Yes: Agent pattern with tool use loop
   -> No: Simple API call or workflow

3. Multi-step, code-orchestrated pipeline
   -> API with tool use - you control the loop

4. Open-ended agent (model decides trajectory)
   -> Agentic loop with tool definitions
```

## Common Patterns

### Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

message = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
print(message.content[0].text)
```

**Streaming** (recommended for long responses):
```python
with client.messages.stream(
    model="claude-sonnet-4-5-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### OpenAI

```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### Ollama (Local)

```python
import ollama

response = ollama.chat(
    model='llama3',
    messages=[{"role": "user", "content": "Hello"}]
)
print(response['message']['content'])
```

Or via OpenAI-compatible API:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # not used but required
)
```

### llama.cpp Server

```python
import requests

response = requests.post("http://localhost:8080/completion", json={
    "prompt": "Hello",
    "n_predict": 128
})
print(response.json()["content"])
```

Or via OpenAI-compatible endpoint:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)
```

## Tool Use / Function Calling

### Anthropic
```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
}]

message = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}]
)
```

### OpenAI
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}]
)
```

## Agentic Loop Pattern

The core pattern for any provider:

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = call_llm(messages)

    if response.has_tool_calls():
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call.name, tool_call.args)
            messages.append(tool_call_message)
            messages.append(tool_result_message(result))
    else:
        # Final response - no more tool calls
        print(response.text)
        break
```

## Common Pitfalls

- Don't truncate inputs silently - notify the user about chunking or summarization
- Don't lowball `max_tokens` - hitting the cap truncates output mid-thought
- Always handle rate limits with exponential backoff
- For streaming, use the SDK's built-in stream helpers rather than raw HTTP
- Parse tool call inputs with proper JSON parsing, not string matching
- For local models, check that the model is loaded before making requests
- Never hardcode API keys - use environment variables or config files

## Prometheus Context

- Use `bash` to test API calls directly with curl
- Use `file_read` and `file_edit` to modify application code
- Use `grep` to find existing API integration patterns in the codebase
- API keys should be in environment variables, never committed (see LCM memory)
- For local model testing, use the Tailscale endpoints in reference_infra
