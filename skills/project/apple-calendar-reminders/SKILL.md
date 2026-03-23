---
name: apple-calendar-reminders
description: Read and write Apple Calendar and Reminders on macOS using the built-in osascript bridge (AppleScript). Use when the user wants to list or add calendar events, list reminder lists, add reminders, or mark reminders complete. Requires local Mac with Calendar and Reminders; first run may prompt for Automation permission.
allowed-tools: execute
metadata:
  platform: macOS
  requires: osascript, Calendar.app, Reminders.app
---

# Apple Calendar & Reminders (osascript)

Use **`execute`** to run **`osascript`** with the patterns below (heredocs recommended). Requires macOS with Calendar and Reminders; use workspace-relative paths under your agent home where applicable.

## Permissions (once per host)

macOS may prompt: **System Settings → Privacy & Security → Automation** — allow your terminal/host (Terminal, iTerm, Cursor, Python) to control **Calendar** and **Reminders**. If scripts return empty errors or “not allowed”, the user must grant access and retry.

Prefer **multi-line** scripts with a **quoted heredoc** so you do not fight shell quoting:

```bash
osascript <<'APPLESCRIPT'
tell application "Calendar"
  name of every calendar
end tell
APPLESCRIPT
```

Escape **single quotes inside** the script as `'\''` if you ever use `osascript -e 'one line'`.

---

## Calendar — read

**List calendar names**

```bash
osascript <<'APPLESCRIPT'
tell application "Calendar"
  name of every calendar
end tell
APPLESCRIPT
```

**Events in a date range** (adjust calendar name `"Home"` to match the user’s calendar; use list from step one)

```bash
osascript <<'APPLESCRIPT'
tell application "Calendar"
  tell calendar "Home"
    set t0 to (current date) - (1 * days)
    set t1 to (current date) + (7 * days)
    set evts to (every event whose start date is greater than or equal to t0 and start date is less than or equal to t1)
    repeat with e in evts
      log (summary of e) & " | " & (start date of e as string) & " | " & (end date of e as string)
    end repeat
  end tell
end tell
APPLESCRIPT
```

Read `log` lines from stderr or switch to `return` a single string by building text in a variable (see write example pattern).

Important correctness notes:
- Use `start date` / `end date` for events. Do not use `date of event` (it is unreliable and often errors).
- Prefer `every event whose start date ...` filtering instead of iterating all events and manually filtering.
- Avoid broad `try ... on error ... end try` that swallows all errors; catch only around optional fields if needed.
- If a command returns empty results unexpectedly, run a minimal diagnostic first (calendar names + bounded query on one known calendar) before retrying large scripts.

**Practical pattern: return one block of text**

```bash
osascript <<'APPLESCRIPT'
set out to ""
tell application "Calendar"
  repeat with c in calendars
    set out to out & "CAL: " & (name of c) & linefeed
  end repeat
end tell
return out
APPLESCRIPT
```

---

## Calendar — write

**Add an event** (set calendar name, summary, start/end as AppleScript date expressions)

```bash
osascript <<'APPLESCRIPT'
tell application "Calendar"
  tell calendar "Home"
    set sd to current date
    set hours of sd to 14
    set minutes of sd to 0
    set ed to sd + (1 * hours)
    make new event at end with properties {summary:"Example from BoyoClaw", start date:sd, end date:ed}
  end tell
end tell
APPLESCRIPT
```

Adjust `Home`, times, and duration. For all-day events, use Calendar’s properties `allday event:true` where supported.

---

## Reminders — read

**List reminder list names**

```bash
osascript <<'APPLESCRIPT'
tell application "Reminders"
  name of every list
end tell
APPLESCRIPT
```

**Incomplete reminders in a list** (replace list name; English default is often `"Reminders"`)

```bash
osascript <<'APPLESCRIPT'
tell application "Reminders"
  tell list "Reminders"
    repeat with r in (every reminder whose completed is false)
      log (name of r) & " | due: " & (due date of r as string)
    end repeat
  end tell
end tell
APPLESCRIPT
```

---

## Reminders — write

**Add a reminder**

```bash
osascript <<'APPLESCRIPT'
tell application "Reminders"
  tell list "Reminders"
    make new reminder with properties {name:"Task from BoyoClaw", body:"Optional notes"}
  end tell
end tell
APPLESCRIPT
```

**Complete a reminder by name** (first match; fragile if duplicates—prefer user confirmation)

```bash
osascript <<'APPLESCRIPT'
tell application "Reminders"
  tell list "Reminders"
    set r to first reminder whose name is "Exact title"
    set completed of r to true
  end tell
end tell
APPLESCRIPT
```

---

## Safety

- Confirm **calendar/list names** with the user when ambiguous.
- Do not delete calendars, lists, or bulk data unless the user explicitly asks.
- AppleScript errors often mean **wrong name**, **no permission**, or **sandbox**—surface the raw `osascript` stderr to the user.
