# Web search

The `web_search` tool runs a chain of search backends and stops at the first
one that answers. With no configuration it uses DuckDuckGo and needs no key
and no setup. A self-hosted SearXNG or a Brave Search key puts a more reliable
backend in front of it.

[← README](../../README.md)

## The lanes

| Lane | What runs | Setup | Use it for |
|---|---|---|---|
| Zero setup | `web_search` → DuckDuckGo | none | a fresh install; works out of the box |
| Your own backends | `web_search` → SearXNG and/or Brave, then DuckDuckGo | `web_search.searxng_url`, or `BRAVE_API_KEY` in the env file | search that does not depend on DuckDuckGo tolerating you |

## How the chain works

```yaml
web_search:
  backends: [searxng, brave, duckduckgo]   # the order they are tried in
  searxng_url: ""                          # e.g. http://127.0.0.1:8888
```

- A backend runs only when it is configured. SearXNG needs `searxng_url`.
  Brave needs `BRAVE_API_KEY` in `~/.config/prometheus/env` or the
  environment. DuckDuckGo always runs.
- Each backend gets one try per search. One that errors, times out
  (15 s for SearXNG and Brave, 20 s per DuckDuckGo endpoint, wall clock), is
  rate-limited, answers with a CAPTCHA or finds nothing falls through to the
  next.
- **The whole search has a 60 s budget.** The backends before DuckDuckGo must
  leave it 30 s, so it always gets its turn. The worst case, every backend
  hanging, ends at 60 s with a message naming each one. That is well inside
  the agent loop's 300 s tool timeout, so the loop never has to cancel a
  search. (httpx's own timeouts are per phase and per read, so a server that
  trickles bytes never trips them; the wall-clock limits are what stop it.)
- **DuckDuckGo is always last.** If you leave it out of `backends`, or list it
  earlier, it is moved to the end and a WARNING says so. It is the fallback
  that needs no setup, so search keeps working when everything else is down.
- You can change the order of the others, for example
  `backends: [brave, searxng, duckduckgo]`.

The result names the backend that answered and anything skipped on the way:

```text
Search results for: llama.cpp grammar (via duckduckgo)
Skipped: searxng (timed out after 15s); brave (rate limited (HTTP 429))
1. llama.cpp/grammars/README.md at master · ggml-org/llama.cpp
   URL: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
   ...
```

With nothing configured, the tool's description is the same text the model
has always been given, byte for byte. With SearXNG or Brave active, the first
sentence names the chain, for example "Search the web via SearXNG (falling
back to DuckDuckGo)". The parameters never change.

## DuckDuckGo: html, then lite, and blocks are reported

DuckDuckGo has two endpoints, `html.duckduckgo.com/html/` and
`lite.duckduckgo.com/lite/`. The tool tries html first and lite second.

When DuckDuckGo suspects a bot it answers with HTTP 202 and a CAPTCHA page
("Unfortunately, bots use DuckDuckGo too."). That page has no results in it.
Before this change the tool reported it as `No search results found.`, which a
model reads as "nothing exists". Now these all count as a **block**:

- a 202, 403 or 429;
- DuckDuckGo's challenge page, whatever the status;
- a 200 page with no results and none of DuckDuckGo's own "No results" markers.

When both endpoints are blocked and nothing earlier in the chain answered, the
tool returns an error that starts `web search is blocked right now:`, says
what DuckDuckGo did, and says it is not an empty result. A genuine empty result
is DuckDuckGo's own "No results" page, and only that is reported as
`No search results found`.

A busy agent can be challenged within a handful of searches. If you see blocks
often, add SearXNG or a Brave key.

## Brave Search

Put the key in the env file, never in `prometheus.yaml`:

```bash
echo 'BRAVE_API_KEY=<your key>' >> ~/.config/prometheus/env
```

Restart the daemon. A key written into `prometheus.yaml` is ignored, with a
WARNING saying where it belongs. The key travels only in Brave's
`X-Subscription-Token` header; it is never in a URL, a result or telemetry,
and that request never follows a redirect (httpx would carry the header along).
A rejected key (401/403) falls through to the next backend.

## SearXNG

SearXNG is a self-hosted metasearch engine. Prometheus uses its JSON API:
`GET <searxng_url>/search?q=…&format=json`.

### The JSON-format gotcha

SearXNG ships with **JSON output disabled**. Asked for `format=json`, a stock
instance answers **403 Forbidden**. Many public instances also put a browser
check in front of the API and answer with an HTML page. In both cases the
tool reports:

```text
SearXNG has JSON output disabled; enable formats: [html, json] (search.formats in its settings.yml)
```

It adds the status, or the HTML page's title, so a bot check can be told apart
from a config problem. Then it falls through to the next backend. Fix it in
SearXNG's `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

### Running one by hand with Docker

Most public instances disable JSON or rate-limit API clients. Run your own:

```bash
mkdir -p ~/.config/searxng
cat > ~/.config/searxng/settings.yml <<'EOF'
use_default_settings: true
server:
  secret_key: "<the output of: openssl rand -hex 32>"
  limiter: false          # the limiter rate-limits API clients like this one
search:
  formats:
    - html
    - json
EOF
docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 \
  -v ~/.config/searxng:/etc/searxng \
  searxng/searxng
```

Check that it answers JSON:

```bash
curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 200
```

Then set `web_search.searxng_url: http://127.0.0.1:8888` in `prometheus.yaml`
and restart the daemon. The boot log names the chain:
`web_search: backends searxng -> duckduckgo`.

## What is recorded

Each call writes one `subsystem_runs` row in `telemetry.db`, with
`subsystem = 'web_search'` and `operation = 'search'`:

| outcome | meaning |
|---|---|
| `success` | the first active backend answered |
| `partial` | a later backend answered after one or more were skipped |
| `failed` | every backend failed (`blocked: true` when DuckDuckGo refused) |

`summary_json` holds the backend that answered, the DuckDuckGo endpoint, the
result count, each skipped backend with its failure kind (`timeout`,
`rate_limited`, `blocked`, `json_disabled`, `no_results` or `error`), and the
inactive backends. It holds no query and no URL. `/health` counts these rows
like any other subsystem's, so a SearXNG that keeps timing out shows up there
as `partial` runs.

```sql
SELECT outcome, json_extract(summary_json, '$.backend') AS backend, COUNT(*)
FROM subsystem_runs WHERE subsystem = 'web_search'
GROUP BY 1, 2;
```
