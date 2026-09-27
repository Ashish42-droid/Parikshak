#!/usr/bin/env python3
"""The project documentation as PDF and Word, from the Markdown in docs/.

    python tools/make_docs.py      # docs/PARIKSHAK_Project_Documentation.{html,pdf,docx}

The Markdown file stays the source of truth; this renders it to a styled HTML
page with markdown-it and asks an installed Microsoft Word to save that as PDF
and DOCX. Without Word, only the HTML is written - it prints cleanly from a
browser too.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 11pt; color: #15202B;
       line-height: 1.45; max-width: 46em; margin: 2em auto; }
h1 { font-family: Arial, sans-serif; font-size: 24pt; color: #0B1D33; margin: 0 0 .2em; }
h2 { font-family: Arial, sans-serif; font-size: 16pt; color: #0B1D33; margin: 1.6em 0 .4em;
     page-break-after: avoid; }
h3 { font-family: Arial, sans-serif; font-size: 12.5pt; color: #2D5BD8; margin: 1.2em 0 .3em;
     page-break-after: avoid; }
p { margin: .45em 0; }
table { border-collapse: collapse; width: 100%; margin: .7em 0 1em; font-size: 10pt;
        page-break-inside: avoid; }
th { background: #0B1D33; color: #FFFFFF; text-align: left; padding: 5px 7px; }
td { border: 1px solid #D3DAE1; padding: 5px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #EEF2F6; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt; background: #EEF2F6;
       padding: 0 2px; }
pre { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #EEF2F6;
      border: 1px solid #D3DAE1; padding: 8px 10px; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
a { color: #2D5BD8; text-decoration: none; }
hr { border: 0; border-top: 1px solid #D3DAE1; margin: 1.4em 0; }
blockquote { border-left: 3px solid #2D5BD8; margin: .8em 0; padding: .1em .9em; color: #4A5866; }
"""


def to_html(md_path: Path, title: str) -> str:
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable("table")
    body = md.render(md_path.read_text(encoding="utf-8"))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>{CSS}</style></head><body>{body}</body></html>')


def word_export(html: Path, pdf: Path, docx: Path) -> None:
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$word = New-Object -ComObject Word.Application\n"
        "$word.Visible = $false\n"
        f"$doc = $word.Documents.Open('{html}', $false, $true)\n"
        f"$doc.SaveAs2('{pdf}', 17)\n"
        f"$doc.SaveAs2('{docx}', 16)\n"
        "$doc.Close($false)\n"
        "$word.Quit()\n")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=300)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the project documentation.")
    ap.add_argument("--source", type=Path, default=ROOT / "docs" / "PROJECT_DOCUMENTATION.md")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "PARIKSHAK_Project_Documentation")
    ap.add_argument("--no-word", action="store_true", help="write the HTML only")
    args = ap.parse_args(argv)

    html = args.out.with_suffix(".html").resolve()
    html.write_text(to_html(args.source, "PARIKSHAK - project documentation"),
                    encoding="utf-8", newline="\n")
    print(f"  {html}")
    if not args.no_word:
        pdf, docx = html.with_suffix(".pdf"), html.with_suffix(".docx")
        word_export(html, pdf, docx)
        print(f"  {pdf}  ({pdf.stat().st_size / 1e3:.0f} KB)")
        print(f"  {docx}  ({docx.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
