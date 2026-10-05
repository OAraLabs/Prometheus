# web_search fixtures

Responses the `web_search` backend tests replay through `httpx.MockTransport`.
No test in CI touches the network. Each file is one of three kinds, and the
tests depend on that difference, so it is written down here.

## Recorded (verbatim bodies, 2026-10-05)

Fetched with `curl -A "Prometheus/0.1"`, the User-Agent the tool sends. Only the
body was kept; the status code is in the file name where it is not 200.

| File | Request | Status |
|---|---|---|
| `ddg_html_results.html` | `html.duckduckgo.com/html/?q=llama.cpp grammar` | 200 |
| `ddg_lite_results.html` | `lite.duckduckgo.com/lite/?q=llama.cpp grammar` | 200 |
| `ddg_lite_anomaly_202.html` | `lite.duckduckgo.com/lite/` a few requests later | **202** |
| `ddg_html_anomaly_202.html` | `html.duckduckgo.com/html/` a few requests later | **202** |
| `searxng_html_bot_check.html` | a public SearXNG instance, `/search?format=json` | 200, `text/html` |

The two 202 pages are DuckDuckGo's real bot challenge ("Unfortunately, bots use
DuckDuckGo too.", an `anomaly-modal` with a duck-picking CAPTCHA). The machine
was challenged on its fifth request across the two endpoints, all within about
a minute, which is how a busy agent gets blocked. The lite page marks result
classes with single quotes (`class='result-link'`); the parser before this
change read double quotes only.

The SearXNG page is what a public instance returns for `format=json` when it
puts a browser check in front of its API: HTML, status 200, no JSON.

## Derived from a recording

| File | How |
|---|---|
| `ddg_html_no_results.html` | `ddg_html_results.html` with the ten results removed and DuckDuckGo's empty-result marker `<div class="no-results">No  results.</div>` (two spaces) in their place |
| `ddg_html_empty_unmarked.html` | the same page with the results removed and **no** marker: a 200 that says nothing, which is how a soft block looks |

The marker text is the one the `duckduckgo_search` library keys on
(`b"No  results."` for html, `b"No more results."` for lite). A real no-results
page could not be recorded: DuckDuckGo was already serving the challenge.

## Constructed from the API's documented shape

No live response was available for these: there is no Docker on the recording
machine for a local SearXNG, and no Brave key.

| File | Shape source |
|---|---|
| `searxng_results.json` | SearXNG's `format=json` output (`results[].url/title/content`, `unresponsive_engines`) |
| `searxng_empty_unresponsive.json` | the same, with no results and two engines reporting why |
| `searxng_403.html` | Werkzeug's stock 403 body, which SearXNG returns (`flask.abort(403)`) for a format missing from `search.formats` |
| `brave_results.json` | Brave Web Search API, `GET /res/v1/web/search` (`web.results[].title/url/description`) |
