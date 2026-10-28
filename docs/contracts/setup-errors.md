# Setup error codes

Every way setup mode can say no has a stable code. Beacon (and any other client) shows a person a plain-English
message and writes the words itself, **from the code**; the English `detail` in a response is for logs, `curl` and
clients that have no copy for a code yet.

## The shape

```json
{"error": "invalid_key", "detail": "a key or token has no control characters or line breaks", "fields": ["api_key"]}
```

- `error` is the code. It is a promise: it keeps its name and HTTP status when the wording changes, and a new
  situation gets a new code instead of reusing one. Clients should treat a code they do not know as a failure of the
  status class it arrives with (4xx: the request; 5xx: the daemon).
- `detail` is English for this occurrence. It can change between versions. Do not parse it.
- `fields` is there when the failure is about particular inputs, so a client can mark the right boxes; it is absent
  when the failure is about the request as a whole. The names are the request's own field names.
- A few codes carry machine data next to these (`attempts_remaining`, `retry_after_seconds`, `reason`).
- Each code has exactly one status. The same code never means a 400 on one route and a 409 on another.

Setup mode is `/api/setup/*` on a daemon with no configuration yet. After setup the full daemon speaks the
vocabulary in the API guide (`docs/guide/api.md`); `GET`/`PUT /api/model` uses `invalid_request` where setup says
`bad_request` (both mean "the request was not what this route expects"), and shares `unsupported_provider`,
`unexpected_base_url`, `invalid_key` and `cloud_key_missing` with setup.

## The codes

| Code | Status | What it means | Fields | Raised by |
|---|---|---|---|---|
| `bad_request` | 400 | The request was not the JSON this route expects, or a field is missing or of the wrong type. | per field, or none | pair, check-key, configure, probe |
| `bad_base_url` | 400 | An address has to start with http:// or https://. | `base_url` | detect, configure |
| `unexpected_base_url` | 400 | A cloud provider is called at its own address and takes none of yours. | `base_url` | check-key, configure |
| `unsupported_provider` | 400 | That provider is not one setup offers. | `provider` | check-key, configure |
| `bad_model` | 400 | The model name has a control character or is far too long. | `model` | configure |
| `bad_target` | 400 | A computer to look at is a name or an address, with an optional port, or an http(s) address with no path. | `target` | probe |
| `cloud_key_missing` | 400 | A cloud provider needs an API key, and none was sent or is already saved. | `api_key` | configure |
| `invalid_key` | 400 | That cannot be an API key: it is empty, far too long, or has a line break. | `api_key` | configure |
| `invalid_token` | 400 | A messaging token has a line break in it or is far too long. | the token's own field | configure |
| `slack_token_pair_incomplete` | 400 | Slack needs both its bot token and its app token, and only one was sent. | `slack_bot_token`, `slack_app_token` | configure |
| `bad_discord_guild_ids` | 400 | Discord server ids are whole numbers. | `discord_guild_ids` | configure |
| `backend_unreachable` | 400 | Nothing at that address answered like a usable inference server of that kind; `reason` says why, in the probe's words (docs/contracts/setup-probe.md). | `base_url` | configure |
| `model_not_served` | 400 | The app at that address answered, and does not serve that model. | `model` | configure |
| `config_write_failed` | 500 | The settings could not be written to disk (permissions, a full disk or a read-only folder). | none | configure |
| `internal_error` | 500 | Something unexpected went wrong inside the daemon; the cause is in its log, not in this answer. | none | any route |
| `not_configured` | 409 | Setup cannot finish yet: there is no valid configuration. | none | complete |
| `setup_mode` | 403 | The daemon is still in setup mode, so only the setup routes exist. | none | every route that is not a setup route |
| `pairing_unavailable` | 403 | A request to join cannot be made until setup has finished. | none | `POST /api/pair/requests` |
| `not_paired` | 401 | Nobody has paired with this daemon yet, so setup calls are refused. | none | detect, check-key, configure, complete |
| `unauthorized` | 401 | The setup token was missing or wrong. | none | detect, check-key, configure, complete |
| `invalid_code` | 401 | The pairing code was wrong. | none (and `attempts_remaining`) | pair |
| `pairing_used` | 403 | The pairing code was already used. | none | pair |
| `pairing_expired` | 403 | The pairing code has expired; restart the daemon for a new one. | none | pair |
| `pairing_locked` | 403 | Too many wrong pairing codes; restart the daemon to try again. | none | pair |
| `not_loopback` | 403 | This pairing secret is only accepted from this Mac, over localhost. | none | pair, with the same-Mac secret |
| `browser_not_allowed` | 400 | This route is for apps, not for a web page. | none | pair, with the same-Mac secret; `/api/hello` |
| `host_not_allowed` | 403 | The daemon only answers requests addressed to localhost while it is bound to this machine. | none | every route, while bound to this machine |
| `rate_limited` | 429 | Too many requests from this address; wait the number of seconds given. | none (and `retry_after_seconds`) | `/api/hello` |

The last four are produced by middleware and routes that setup mode shares with the full daemon; they are listed so
a client sees the whole vocabulary.

## Keeping this true

`src/prometheus/web/setup_errors.py` is the source: `CATALOG` holds the codes and statuses, and `refuse()` is the only
way `setup_server.py` answers an error (a test fails on any hand-built error response). `tests/test_setup_error_codes.py`
fails when this table and the catalog disagree on a code or a status, and when a code in the catalog is exercised by
no test.
