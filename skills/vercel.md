---
name: vercel
description: "Vercel CLI workflow — login (browser auth), `vercel` for preview / `vercel --prod` for production, list deployments, inspect logs, manage env vars and domains. Use when the user asks to deploy to Vercel, pull Vercel env vars, check Vercel deployment status, or set up a Vercel domain."
license: MIT
allowed-tools: "Bash"
---

# Vercel CLI — Full Workflow Skill

## Authentication
When the user wants to use Vercel for the first time or gets auth errors:
1. Run: `vercel login`
   - This opens a browser — tell user to click Continue
2. Verify: `vercel whoami`

## Common Commands
vercel                          # deploy to preview URL
vercel --prod                   # deploy to production
vercel ls                       # list recent deployments
vercel inspect <url>            # deployment details
vercel logs <url>               # deployment logs
vercel env ls                   # list env vars
vercel env add KEY              # add env var (interactive)
vercel env pull .env.local      # pull env vars to local
vercel domains ls               # list domains
vercel link                     # link current dir to project

## Troubleshooting
- `Error: not authenticated` → run `vercel login`
- `Error: not linked` → run `vercel link`
- Build fails → run `vercel logs <deployment-url>`
