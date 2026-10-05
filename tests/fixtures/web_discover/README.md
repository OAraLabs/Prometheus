# web_discover fixtures

Exa API responses the `web_discover` tests replay through an
`httpx.MockTransport`. No test calls Exa.

**These are constructed, not recorded.** There was no Exa key on the machine
that wrote them. Each follows the shapes in Exa's published OpenAPI spec
(`https://exa.ai/docs/exa-spec.yaml`, fetched 2026-10-05):

| File | Endpoint | Shape |
|---|---|---|
| `exa_search_results.json` | `POST /search` | `SearchResultsResponse`: `requestId`, `results[]` (`title`, `url`, `id`, `publishedDate`, `author`, `summary`), `costDollars`, `searchTime` |
| `exa_find_similar_results.json` | `POST /findSimilar` | `FindSimilarResponse`: the same, plus a per-result `score` |
| `exa_error_401.json` | either, bad key | `ErrorResponse`: `requestId`, `error`, `tag` |
| `exa_error_402.json` | either, out of credits | `ErrorResponse` |

The last result in `exa_search_results.json` has no `summary`, and the third
one's summary is longer than the tool's cap. Both are on purpose.
Replace these with recordings when a key is available. The tests assert on the
fields, not on the bytes, so a recording that keeps the documented shape will
still pass.
