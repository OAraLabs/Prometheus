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
| Your own backends | `web_search` → SearXNG and/or Brave, then DuckDuckGo | `oara search setup` (SearXNG in Docker), or `BRAVE_API_KEY` in the env file | search that does not depend on DuckDuckGo tolerating you |

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
  (15 s for SearXNG and Brave, 20 s per DuckDuckGo endpoint), is rate-limited,
  answers with a CAPTCHA or finds nothing falls through to the next.
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

### `oara search setup`

Most public instances disable JSON or rate-limit API clients, so run your own.
One command does it, given Docker:

```bash
oara search setup --dry-run   # what it would do; changes nothing
oara search setup             # do it (default port 8888)
oara search setup --port 8890
```

(`prometheus search setup` is the same command under the old name.) In order:

1. It finds the config file the daemon reads. With none it stops and says to
   run `oara setup` first.
2. It checks for Docker and that its daemon answers. Without Docker it says
   so, points here for the manual route, and **changes nothing**.
3. It looks for a container named `searxng` (`--name` to change it). A running
   one is reused and a stopped one is started. A container of that name running
   some other image is refused. With none, it writes
   `~/.prometheus/searxng/settings.yml` (JSON output on, the limiter off, a
   random `secret_key`) and runs `searxng/searxng` with
   `--restart unless-stopped`, published on `127.0.0.1:<port>` only.
4. It waits (up to `--timeout`, default 90 s) until
   `/search?q=test&format=json` answers JSON. It uses the same check as
   web_search, so a 403 or an HTML answer gets the JSON-disabled message above.
5. It backs up `prometheus.yaml` (`prometheus.yaml.bak`, or `.bak.1`, `.bak.2`,
   … so no earlier backup is overwritten) and sets
   `web_search.searxng_url: "http://127.0.0.1:<port>"`. Only that key changes.
   Comments and every other key stay as they were, and the edit is checked by
   re-parsing before it is written.

Running it again is safe. It finds the container, sees the URL is already set,
and changes nothing. If something fails on the way, the config is left alone.
Restart the daemon afterwards; its boot log then names the chain.

The container gets `FORCE_OWNERSHIP=false`. Without it, the image's entrypoint
(running as root) `chown`s the mounted `/etc/searxng` to its own user. On Linux
that would leave `~/.prometheus/searxng` owned by a uid you don't have. The
settings file is written world-readable (0644) instead, so the container can
read it without owning it.

### Running one by hand with Docker

Without `oara search setup`, the same thing by hand:

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
  -e FORCE_OWNERSHIP=false \
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
