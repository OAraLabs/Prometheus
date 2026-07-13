---
name: web-search-fetch
description: Search the web and fetch content from URLs using web_search and web_fetch tools. Use for researching topics, reading documentation, checking API references, fetching web pages, accessing X/Twitter content, or any task requiring live web information. Replaces CLI tools like xurl with native web access.
---

# Web Search and Fetch

Use the `web_search` and `web_fetch` tools to access live web information.

## Tools Available

### web_search

Search the web for current information. Returns search results with titles, URLs, and snippets.

Use for:
- Researching a topic or finding documentation
- Finding recent news or updates
- Discovering APIs, libraries, or tools
- Looking up error messages or solutions
- Finding social media posts or discussions

### web_fetch

Fetch the content of a specific URL. Returns the page content as text.

Use for:
- Reading documentation pages
- Fetching API responses
- Accessing specific web pages
- Reading blog posts or articles
- Getting raw content from known URLs

## Common Workflows

### Research a Topic

1. Use `web_search` with a focused query
2. Review results for relevance
3. Use `web_fetch` on the most promising URLs
4. Synthesize findings

### Check API Documentation

1. `web_search` for "[service] API documentation"
2. `web_fetch` the official docs URL
3. Extract relevant endpoints, parameters, auth methods

### Find Social Media Content

For X/Twitter posts:
1. `web_search` for "site:x.com [query]" or "site:twitter.com [query]"
2. `web_fetch` on specific post URLs

For other platforms, use similar site-scoped searches.

### Monitor a Topic

1. `web_search` with time-relevant queries
2. Compare with previously known information
3. Report new findings

### Fetch and Process Data

1. `web_fetch` a known data URL (JSON API, CSV endpoint, etc.)
2. Process the returned content
3. Write results via `file_write` if needed

## Tips

- Use specific, focused search queries for better results
- Add "site:domain.com" to restrict searches to specific sites
- Add date-related terms for recency ("2026", "latest", "recent")
- Use `web_fetch` directly when you already know the URL
- For APIs that return JSON, `web_fetch` the endpoint and parse the response
- Chain searches: start broad, then narrow based on initial results

## X/Twitter Workflows

Instead of a dedicated CLI tool, use web search and fetch for X/Twitter:

- **Read a post**: `web_fetch` the post URL
- **Search posts**: `web_search` with "site:x.com [query]"
- **User profile**: `web_search` for "site:x.com [username]"
- **Trending topics**: `web_search` for "trending on X" or "trending on Twitter"

For posting or interacting with X/Twitter programmatically, use `bash` with `curl` and the X API v2 directly (requires authentication configured outside the agent session).

## Error Handling

- If `web_fetch` fails, the URL may be behind authentication or rate-limited
- Try alternative URLs from `web_search` results
- For dynamic pages that do not render well, try adding `?format=text` or look for API endpoints
- Some sites block automated access -- search for cached or mirrored versions
