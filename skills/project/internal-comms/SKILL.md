---
name: internal-comms
description: Draft internal workplace communications—status updates, 3P (Progress/Plans/Problems), leadership summaries, incident notes, FAQs, newsletters, project updates. Use when tone is professional, concise, and audience is teammates or org-wide. Gather facts first; avoid inventing metrics.
allowed-tools: execute read_file write_file edit_file grep glob ls
metadata:
  upstream: https://github.com/anthropics/skills/tree/main/skills/internal-comms
  license_note: Summarized from Anthropic public skills repo; adapt tone to the user's org.
---

# Internal communications

## Pick a shape
- **3P update**: Progress (done), Plans (next), Problems (blockers + asks). Short bullets; owners and dates when known.
- **Status report**: What shipped, what slipped, risks, next milestone.
- **Incident**: timeline, impact, root cause (if known), mitigation, follow-ups—no blame tone.
- **FAQ**: question as heading, crisp answer, link to source doc if any.
- **Newsletter**: sections with headlines; mix wins, links, lightweight calls to action.

## Process
1. Confirm audience and sensitivity (internal-only vs leadership).
2. Ask for or infer **facts**; label estimates clearly.
3. Draft; keep paragraphs short; front-load the takeaway.
4. Offer optional shorter “TL;DR” blurb for email/chats.

## Style
- Clear subject lines / titles.
- Active voice; specific dates and owners.
- If policies or legal could apply, remind the user to verify with their team.
