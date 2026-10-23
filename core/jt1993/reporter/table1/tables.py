"""Display-ready Table I with the paper's panel, horizon, and portfolio order."""

import csv
from html import escape
from pathlib import Path

import pandas as pd


def table1_rows(summary):
    """Build two header rows and 24 data rows, matching the supplied paper table.

    Panels A/B appear side by side. Within each panel K is 3, 6, 9, 12.
    For each J=3,6,9,12, Sell, Buy, and Buy-sell each have a return row and
    a following parenthesized t-statistic row. Blank columns carry the paper's
    two 'K =' header labels. Use calculated mean/t, never benchmark columns.
    """
    horizons = (3, 6, 9, 12)
    portfolios = (("loser", "Sell"), ("winner", "Buy"), ("buy_sell", "Buy-sell"))
    keys = ["panel", "J", "K", "leg"]
    expected = pd.MultiIndex.from_product(
        [("A", "B"), horizons, horizons, [leg for leg, _ in portfolios]], names=keys,
    )
    values = summary.set_index(keys)
    if not values.index.is_unique or set(values.index) != set(expected):
        raise ValueError("Table I needs exactly one calculated mean/t for every panel/J/K/portfolio.")
    rows = [
        ["", "", "Panel A", "", "", "", "", "Panel B", "", "", "", ""],
        ["J", "", "K =", "3", "6", "9", "12", "K =", "3", "6", "9", "12"],
    ]
    for J in horizons:
        for leg, label in portfolios:
            means, tstats = [str(J), label], ["", ""]
            for panel in ("A", "B"):
                means.append("")
                tstats.append("")
                for K in horizons:
                    result = values.loc[(panel, J, K, leg)]
                    means.append(f"{result['mean']:.4f}" if pd.notna(result["mean"]) else "NA")
                    tstats.append(f"({result['t']:.2f})" if pd.notna(result["t"]) else "(NA)")
            rows.extend([means, tstats])
    return rows


def write_table1(summary, output_dir, caption=""):
    """Write the primary CSV and a plain HTML table; return paths and HTML markup."""
    rows = table1_rows(summary)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)

    body = []
    for i, row in enumerate(rows[2:]):
        kind = "mean-row" if i % 2 == 0 else "t-row"
        cells = "".join(f"<td>{escape(value)}</td>" for value in row)
        body.append(f'<tr class="{kind}">{cells}</tr>')
    subheader = "".join(f"<th>{escape(value)}</th>" for value in rows[1][2:])
    markup = f'''<style>
.table1-report{{font:18px/1.25 "Times New Roman",serif;color:#111;background:white;padding:22px;margin:0}}
.table1-report .table-title{{text-align:center;font-weight:bold;font-size:22px;margin:0 0 5px}}
.table1-report .table-subtitle{{text-align:center;margin:0 0 16px}}
.table1-report .table-scroll{{overflow-x:auto}}
.table1-report table{{border-collapse:collapse;margin:auto;min-width:920px;border-top:2px solid #111;border-bottom:2px solid #111}}
.table1-report th{{font-weight:normal;text-align:center;padding:5px 12px}}
.table1-report thead{{border-bottom:1px solid #111}}
.table1-report thead tr:first-child th[colspan]{{border-bottom:1px solid #111;font-size:21px}}
.table1-report td{{text-align:right;padding:0 12px;white-space:nowrap;font-variant-numeric:tabular-nums}}
.table1-report td:nth-child(2){{text-align:left;min-width:85px}}
.table1-report td:nth-child(3),.table1-report td:nth-child(8){{min-width:20px}}
.table1-report .mean-row td{{padding-top:8px}}
.table1-report .t-row td{{padding-bottom:2px}}
.table1-report .table-notes{{font-size:15px;text-align:center;margin:12px 0 0}}
</style><section class="table1-report">
<p class="table-title">Table I</p><p class="table-subtitle">Returns of Relative Strength Portfolios</p>
<div class="table-scroll"><table aria-label="Replicated Table I">
<thead><tr><th rowspan="2"><i>J</i></th><th rowspan="2"></th><th colspan="5">Panel A</th><th colspan="5">Panel B</th></tr>
<tr>{subheader}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<p class="table-notes">{escape(caption)}<br>Calculated monthly returns in decimals; t-statistics in parentheses.</p>
</section>'''
    html_path = output_dir / "table1.html"
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>JT1993 replication - Table I</title></head><body>' + markup + '</body></html>',
        encoding="utf-8",
    )
    return {"table1_csv": csv_path, "table1_html": html_path}, markup
