---
name: pdf-work
description: Read, merge, split, create, or extract from PDFs using Python and CLI tools. Use when the user mentions .pdf, wants text or tables extracted, pages rotated or merged, new PDFs generated, watermarks, OCR, or password protection. Triggers include "merge PDFs", "extract tables from PDF", "fill PDF", "OCR scan".
allowed-tools: execute read_file write_file edit_file grep glob ls
metadata:
  upstream: https://github.com/anthropics/skills/tree/main/skills/pdf
  license_note: Summarized from Anthropic public skills repo; install pypdf pdfplumber reportlab as needed.
---

# PDF work (real-world)

## Stack (install as needed)
- **pypdf**: merge, split, rotate, encrypt, basic read
- **pdfplumber**: text + tables with layout
- **reportlab**: generate PDFs (canvas / platypus)
- **CLI**: `pdftotext`, `qpdf`, `pdfimages` (poppler) when available

## Patterns

**Extract text (pypdf)**  
Use `PdfReader`, iterate `pages`, `extract_text()`. For layout-heavy docs prefer pdfplumber.

**Tables (pdfplumber)**  
`page.extract_tables()` → optional `pandas.DataFrame` → export `.xlsx` or `.csv` via `execute` + short Python.

**Merge / split**  
`PdfWriter` + `add_page` from multiple `PdfReader`s; or `qpdf --empty --pages a.pdf b.pdf -- out.pdf`.

**Create PDF**  
reportlab `Canvas` or `SimpleDocTemplate`; avoid Unicode subscript chars in ReportLab—use `<sub>` / `<super>` in `Paragraph` XML.

**Scanned / OCR**  
`pdf2image` + `pytesseract` if installed; else suggest the user run OCR externally.

**Forms**  
Form filling varies by PDF; use pypdf 3.x form APIs or specialized libs; test on a copy of the file inside the workspace.

## Workspace
Keep inputs/outputs under paths the user provides; use `read_file` to inspect extracted `.txt` or small scripts before running.
