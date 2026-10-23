"""Shared offline table panels and embedded figures for the submission."""

import base64
from html import escape
from io import BytesIO
import re


def html_table(headers, rows, label, *, groups=(), t_rows=False):
    region_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    group_header = ""
    if groups:
        group_header = "<tr>" + "".join(
            f'<th scope="colgroup" colspan="{width}">{escape(title)}</th>' for title, width in groups
        ) + "</tr>"
    body = []
    for i, row in enumerate(rows):
        kind = ' class="t-row"' if t_rows and i % 2 else ""
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body.append(f"<tr{kind}>{cells}</tr>")
    return (f'<div class="table-panel"><div class="table-toolbar"><p>{escape(label)}</p>'
            f'<button type="button" class="table-expand" aria-expanded="false" aria-controls="{region_id}">Expand table</button></div>'
            f'<div id="{region_id}" class="table-scroll" tabindex="0" role="region" aria-label="{escape(label)}">'
            f'<table><caption class="sr-only">{escape(label)}</caption><thead>{group_header}<tr>'
            + "".join(f'<th scope="col">{escape(title)}</th>' for title in headers)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></div>")



def figure_html(figure, caption):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, facecolor="white", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}">'
            f"<figcaption>{escape(caption)}</figcaption></figure>")

