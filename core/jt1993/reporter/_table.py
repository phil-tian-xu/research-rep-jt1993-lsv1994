"""Plain CSV/HTML output shared by the Table II and VII reporters."""

import csv
from html import escape


def write_display_table(rows, path_prefix, title, notes):
    """Save literal display strings with one header row and no extra CSV index."""
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, html_path = path_prefix.with_suffix(".csv"), path_prefix.with_suffix(".html")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    header = "".join(f"<th>{escape(value)}</th>" for value in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows[1:])
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{font:18px/1.3 "Times New Roman",serif;color:#111;background:white;margin:30px}}
h1{{font-size:23px;text-align:center}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;margin:auto;border-top:2px solid;border-bottom:2px solid}}
th{{font-weight:normal;border-bottom:1px solid;padding:6px 18px;white-space:pre-line}}
td{{text-align:right;padding:2px 18px;white-space:nowrap;font-variant-numeric:tabular-nums}}
td:first-child{{text-align:left}}p{{font-size:15px;text-align:center}}
</style><h1>{escape(title)}</h1><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div><p>{escape(notes)}</p></html>'''
    html_path.write_text(html, encoding="utf-8")
    print(f"Display CSV: {csv_path}")
    print(f"Display table: {html_path}")
    return {"csv": csv_path, "html": html_path}
