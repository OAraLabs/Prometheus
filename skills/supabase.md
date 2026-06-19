---
name: supabase
description: "Supabase CLI workflow — login (browser auth), link projects, run migrations (`db push` / `db pull` / `db diff`), deploy edge functions, manage secrets, generate TypeScript types, run the local stack. Use when the user asks to push/pull a Supabase migration, deploy a Supabase edge function, manage Supabase secrets, or work with a linked Supabase project."
license: MIT
allowed-tools: "Bash"
---

# Supabase CLI — Full Workflow Skill

## Authentication
When the user wants to use Supabase for the first time or gets auth errors:
1. Run: `supabase login`
   - This opens a browser — tell user to click Authorize CLI
2. Verify: `supabase projects list`

## Common Commands
supabase init                           # initialize in current dir
supabase link --project-ref <ref>       # link to remote project
supabase status                         # local dev status
supabase db push                        # push migrations to remote
supabase db pull                        # pull remote schema
supabase db diff                        # diff local vs remote
supabase migration new <name>           # create new migration
supabase migration list                 # list migrations
supabase functions deploy <name>        # deploy edge function
supabase functions serve                # serve locally
supabase secrets list                   # list secrets
supabase secrets set KEY=value          # set a secret
supabase gen types typescript --linked  # generate TS types
supabase start                          # start local stack
supabase stop                           # stop local stack

## Troubleshooting
- `Error: not logged in` → run `supabase login`
- `Error: project not linked` → run `supabase link --project-ref <ref>`
- Local stack won't start → Docker must be running
- Find project ref: Dashboard → Settings → General → Reference ID
