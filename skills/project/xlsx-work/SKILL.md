---
name: xlsx-work
description: Create, edit, analyze Excel .xlsx/.xlsm or tabular CSV/TSV. Use for spreadsheets as deliverable—formulas, formatting, cleaning data, charts, multi-sheet work. Triggers include spreadsheet by name, "Excel model", pivot, .csv cleanup. Prefer formulas in cells over hardcoding computed numbers from Python.
allowed-tools: execute read_file write_file edit_file grep glob ls
metadata:
  upstream: https://github.com/anthropics/skills/tree/main/skills/xlsx
  license_note: Summarized from Anthropic public skills repo; requires pandas and/or openpyxl.
---

# Spreadsheet work (real-world)

## Libraries
- **pandas**: load/analyze/export, bulk transforms, CSV/TSV
- **openpyxl**: formulas, formatting, existing workbook structure—preserve templates when editing

## Rules
- **Formulas in Excel**, not Python results pasted as static values, whenever the sheet must stay editable (totals, growth %, links).
- After writing formulas with openpyxl, **recalculate** if your environment has LibreOffice (`soffice --headless --calc --convert-to ...`) or open in Excel; note that `data_only=True` load **destroys formulas** if saved—avoid unless intentionally materializing values.
- **Financial style** (unless user/template says otherwise): blue inputs, black formulas, green same-workbook links, red external links, yellow key assumptions; document hardcodes with source comments.
- **Zero formula errors** before delivery: scan for `#REF!`, `#DIV/0!`, `#VALUE!`.

## Workflow
1. Inspect with pandas (`head`, `info`) or openpyxl `load_workbook`.
2. Modify: `openpyxl` for styles/formulas; pandas for data surgery then `to_excel`.
3. Multi-sheet: iterate `wb.sheetnames`.
4. Verify column letters vs 1-based indices; Excel rows are 1-indexed.

## Snippets
- `pd.read_excel(path, sheet_name=None)` for all sheets dict.
- `openpyxl.load_workbook(path)` then `ws['A1']` or named sheet.
- Insert/delete rows/columns with openpyxl helpers when restructuring.
