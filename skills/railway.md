---
name: railway
description: "Railway CLI workflow — login (browser auth), deploy via `railway up`, view logs, manage env vars and services. Use when the user asks to deploy to Railway, check Railway logs, set Railway env vars, or troubleshoot Railway authentication."
license: MIT
allowed-tools: "Bash"
---

# Railway CLI — Full Workflow Skill

## Authentication
When the user wants to use Railway for the first time or gets auth errors:
1. Run: `railway login`
   - This opens a browser on the user's LOCAL machine (not the server)
   - Tell the user: "A browser window should open — click Accept to authenticate"
   - Wait for them to confirm they clicked accept
2. Verify: `railway whoami`
3. If browser doesn't open, try: `railway login --browserless` (gives a URL to paste)

Token stored at: `~/.railway/config.json`

## Common Commands
railway status                  # current project + environment
railway logs                    # tail live logs
railway logs --tail 100         # last 100 lines
railway up                      # deploy current directory
railway variables               # list all env vars
railway variables set KEY=value # set a variable
railway service                 # list services
railway environment             # list/switch environments
railway link                    # link to a project

## Troubleshooting
- `Error: not linked` → run `railway link` first
- `Error: unauthorized` → run `railway login` again
- Deployment stuck → check `railway logs`
