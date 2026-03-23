---
name: docx-work
description: Create or edit Word .docx files—reports, memos, letters, templates. Use when deliverable is .docx or user asks for Word-compatible formatting. For heavy layout use docx npm library or python-docx; for text extraction pandoc is ideal. Not for Google Docs or raw PDF as primary output.
allowed-tools: execute read_file write_file edit_file grep glob ls
metadata:
  upstream: https://github.com/anthropics/skills/tree/main/skills/docx
  license_note: Summarized from Anthropic public skills repo; full upstream includes unpack/pack scripts—use pandoc or docx-js when possible.
---

# DOCX work (real-world)

## Facts
- `.docx` is a ZIP of XML; prefer **high-level tools** before hand-editing XML.

## Reading / convert
- **pandoc**: `pandoc in.docx -o out.md` (or `--track-changes=all` when reviewing edits).
- **python-docx**: good for paragraphs, tables, styles without raw XML.

## Creating new documents
- **Node `docx` package** (`npm i docx`): programmatic docs with sections, headings, tables; set **page size explicitly** (US Letter vs A4) and margins in DXA if using that API.
- **python-docx**: simpler documents; limited vs full OOXML.

## Editing existing
- Prefer **load → change paragraphs/tables → save** with python-docx.
- If you must patch OOXML, unzip to a folder, edit `word/document.xml` carefully, re-zip—error-prone; only when necessary.

## Quality
- No fake “unicode bullets” for numbered lists—use real list/numbering features of the library.
- Tables: set stable column widths; avoid percentage widths if compatibility matters.
- Images: embed with correct content-type; prefer PNG/JPEG.

## Legacy .doc
Convert to `.docx` first (LibreOffice `soffice --headless --convert-to docx` if installed).
