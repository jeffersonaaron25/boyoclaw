---
name: url-reader
description: "Fetches and extracts the key content from any URL. Use when user shares a link, asks to read a page, extract text from a URL, or pull content from a website. Efficiently extracts main text while stripping ads, navigation, and boilerplate. Trigger words: read this URL, fetch this page, extract from URL, pull content from, open this link"
allowed-tools: execute,read_file,write_file,edit_file,grep,glob,ls,send_message
---

# URL Reader — Fetch & Extract

## When to Use This Skill
Use when the user wants content extracted from a specific URL, or when you need to read a webpage to answer a question.

## How It Works
1. **Validate the URL** — confirm it's a valid, accessible web address
2. **Fetch the content** — use `execute` with `curl -L -s --max-time 30` to get the HTML
3. **Strip boilerplate** — use `lynx -dump -nolist` or `sed`/`grep` to extract visible text
4. **Clean up** — remove excess whitespace, ads, navigation menus
5. **Return the content** — present clean text to the user

## Commands
```bash
# Basic fetch
curl -L -s --max-time 30 "https://example.com"

# Fetch and strip HTML tags (if lynx available)
lynx -dump -nolist "https://example.com"

# Fetch with user-agent (for sites that block bots)
curl -L -s --max-time 30 -A "Mozilla/5.0" "https://example.com"

# Fetch and extract article text using readability-like approach
curl -L -s --max-time 30 "https://example.com" | sed 's/<[^>]*>//g' | sed '/^$/d'
```

## Output Format
```
## [Page Title]
**URL:** [original URL]
**Fetched:** [timestamp]

### Content
[Clean, readable text extracted from the page]

### Notes
[Any issues with the fetch — paywall, blocked, JS-heavy, etc.]
```

## Tips
- Always include the original URL in your response
- Note if content was partially blocked or inaccessible
- For JS-heavy sites (SPAs), note that content may be limited
- Strip email addresses and phone numbers if the user is privacy-conscious
