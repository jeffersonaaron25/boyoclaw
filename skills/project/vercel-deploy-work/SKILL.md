---
name: vercel-deploy-work
description: Deploy a web app or static site to Vercel using the Vercel CLI. Use when the user asks to deploy, get a preview URL, ship a branch, or go live. Default to preview deployments unless they explicitly ask for production.
allowed-tools: execute read_file write_file edit_file grep glob ls
metadata:
  upstream: https://github.com/openai/skills/tree/main/skills/.curated/vercel-deploy
  license_note: Summarized from OpenAI skills catalog; requires Vercel CLI and user auth.
---

# Vercel deploy (CLI)

## Before you run
- Check `vercel --version` or `command -v vercel`.
- Deploys need **network** and often **auth** (`vercel login` may be required)—tell the user if credentials are missing.
- Prefer **preview**: `vercel deploy [path] -y` (or project root `.`).
- Use **production** only if the user clearly asks: `vercel deploy --prod -y`.

## Flow
1. `cd` to the project root that contains `package.json` or framework config.
2. Run deploy with a generous timeout (builds can take many minutes).
3. Return the **preview/production URL** from CLI output; do not claim the site is “verified live” unless the user checks it.

## Monorepos
Pass the app subdirectory if needed: `vercel deploy ./apps/web -y`.

## Troubleshooting
- Build failures: read the CLI log; fix missing env vars (`vercel env pull` pattern) or build errors locally first.
- If sandbox blocks outbound network, the user must run deploy outside the sandbox or grant network access.
