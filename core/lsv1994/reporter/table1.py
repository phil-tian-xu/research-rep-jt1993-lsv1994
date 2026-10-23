"""Display Table I and comparisons using saved results only."""

import base64
from html import escape
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.lsv1994.replication._formation import SIGNALS
from core.lsv1994.replication.table1 import STATS, summarize_table1, validate_cohort_coverage
from utils.path_info import root_path


LABELS = {"B/M": "Panel A: Book-to-market", "C/P": "Panel B: Cash-flow-to-price",
          "E/P": "Panel C: Earnings-to-price", "GS": "Panel D: Past sales growth"}


def report_table1(result_dir, *, show=False):
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    for name, digest in info["output_sha256"].items():
        if hashlib.sha256((result_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved output changed or belongs to a different run: {name}")
    summary = pd.read_csv(result_dir / "table1_summary.csv")
    yearly = pd.read_csv(result_dir / "portfolio_year_returns.csv")
    validate_cohort_coverage(yearly, info["formation_start"], info["formation_end"], info["return_end"])
    calculated = summarize_table1(yearly).set_index(["strategy", "stat", "decile"])["return"]
    saved = summary.set_index(["strategy", "stat", "decile"])["return"]
    if not saved.index.is_unique or set(saved.index) != set(calculated.index):
        raise ValueError("Invalid Table I summary keys.")
    if not np.allclose(saved.reindex(calculated.index), calculated, atol=1e-12, rtol=1e-10):
        raise ValueError("Table I summary disagrees with saved cohort returns.")
    validation_text = "Independent notebook replay has not been recorded for these saved outputs."
    validation_path = result_dir / "reference_validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256((result_dir / "table1_summary.csv").read_bytes()).hexdigest()
        if (validation.get("passed") and validation.get("summary_sha256") == digest
                and validation.get("notebook_sha256") == info["reference_notebook_sha256"]):
            validation_text = (f'Independent notebook replay: {validation["cells_compared"]} cells match; '
                               f'maximum absolute return difference {validation["maximum_absolute_return_difference"]:.2g}.')
    currencies = info["diagnostics"]["formation_account_currencies"]
    foreign = sum(n for code, n in currencies.items() if code not in ("USD", "missing"))
    currency_note = (f'<p class="notice">Reference convention: {foreign:,} formation observations have non-USD accounting values, '
                     'with no currency conversion before dividing by USD market equity. These ratios are not currency-consistent. '
                     'A separate USD-only sensitivity is available in main_lsv.py.</p>') if foreign else ""
    report_dir = result_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    table = summary.pivot(index=["strategy", "stat"], columns="decile", values="return").reindex(
        pd.MultiIndex.from_product([SIGNALS, STATS], names=["strategy", "stat"]))
    table.to_csv(report_dir / "table1.csv")
    # Write the complete comparisons in machine-readable form, keeping notebook
    # output and published paper as separate benchmarks.
    summary.to_csv(report_dir / "table1_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, strategy in zip(axes.flat, SIGNALS):
        ar = summary.loc[summary["strategy"].eq(strategy) & summary["stat"].eq("AR")].sort_values("decile")
        ax.plot(ar["decile"], ar["return"] * 100, "o-", label="Replication", color="#176b80")
        if "paper" in ar:
            ax.plot(ar["decile"], ar["paper"] * 100, "s--", label="Published paper", color="#a46330")
        ax.set(title=LABELS[strategy], xlabel="Ascending signal decile", ylabel="Average annual return (%)", xticks=range(1, 11))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    ar_path = report_dir / "average_annual_returns.png"
    fig.savefig(ar_path, dpi=160)
    if not show:
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    r1 = yearly.loc[yearly["hold_year"].eq(1)]
    spreads = {}
    for strategy in SIGNALS:
        pivot = r1.loc[r1["strategy"].eq(strategy)].pivot(index="formation_year", columns="portfolio", values="ret")
        glamour, value = (10, 1) if strategy == "GS" else (1, 10)
        spreads[strategy] = pivot[value] - pivot[glamour]
        ax.plot(pivot.index, spreads[strategy] * 100, marker="o", markersize=3, label=strategy)
    ax.axhline(0, color="#666666", linewidth=.8)
    ax.set(title="First holding-year return: value minus glamour", xlabel="April formation year",
           ylabel="Annual return spread (percentage points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4)
    spread_path = report_dir / "r1_value_minus_glamour.png"
    fig.savefig(spread_path, dpi=160)
    pd.DataFrame(spreads).to_csv(report_dir / "r1_value_minus_glamour.csv")
    if not show:
        plt.close(fig)
    sections = []
    comparisons = []
    for strategy in SIGNALS:
        panel = table.loc[strategy].reindex(STATS)
        label = "Decile 1 is value; decile 10 is glamour." if strategy == "GS" else "Decile 1 is glamour; decile 10 is value."
        sections.append(f'<section><h2>{escape(LABELS[strategy])}</h2><p>{label}</p>'
                        + panel.to_html(float_format=lambda v: f"{v:.4f}", border=0) + '</section>')
        if "paper" in summary:
            x = summary.loc[summary["strategy"].eq(strategy)]
            paper = x.pivot(index="stat", columns="decile", values="paper").reindex(STATS)
            diff = x.pivot(index="stat", columns="decile", values="difference_paper_pp").reindex(STATS)
            comparisons.append(f'<details><summary>{escape(LABELS[strategy])}: paper and differences</summary>'
                               '<h3>Published paper (decimal returns)</h3>'
                               + paper.to_html(float_format=lambda v: f"{v:.3f}", border=0)
                               + '<h3>Replication minus paper (percentage points)</h3>'
                               + diff.to_html(float_format=lambda v: f"{v:.2f}", border=0) + '</details>')
    figures = []
    for image_path, label in [(ar_path, "Average annual returns by decile"),
                              (spread_path, "First holding-year value-minus-glamour spreads")]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        figures.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')
    counts = yearly.groupby(["strategy", "hold_year"])["formation_year"].nunique().unstack().reindex(SIGNALS)
    notes = ''.join(f'<li>{escape(note)}</li>' for note in info["limitations"])
    diagnostics = pd.Series(info["diagnostics"], name="Value").to_frame().to_html(border=0)
    caption = (f'{info["universe"]} | April formations {info["formation_start"]}-{info["formation_end"]}'
               f' | holding returns end {info["return_end"]}')
    run_plan = ('<details><summary>Requested sample and effective coverage</summary><pre>'
                + escape(json.dumps(info["run_plan"], indent=2)) + '</pre></details>') if info.get("run_plan") else ""
    markup = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LSV1994 Table I</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;background:#f4f6f7;color:#24323e;margin:0;padding:28px}}
main{{max-width:1200px;margin:auto}}section,details,figure{{background:white;padding:20px;margin:18px 0;border:1px solid #dce1e6;border-radius:6px;overflow:auto}}
h1,h2,h3{{line-height:1.25}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid #e3e7ea}}thead th{{background:#eef3f5}}
tbody th{{text-align:left}}img{{display:block;width:100%;height:auto}}summary{{cursor:pointer;font-weight:600}}.notice{{background:#fff3d9;padding:14px;border-left:4px solid #b97616}}figcaption{{color:#566370}}
</style></head><body><main><h1>LSV1994: Table I</h1><p>{escape(caption)}</p>
<p>{escape(validation_text)}</p>{currency_note}{run_plan}
<p>All table returns are decimals: 0.1000 means 10%. R1-R5 average annual portfolio returns across available formation cohorts.
AR averages the five horizon means; CR5 compounds those means. SAAR is the average annual size-adjusted return.</p>
{''.join(sections)}<section><h2>Cohort coverage</h2>{counts.to_html(border=0)}</section>
{''.join(figures)}<h2>Published-paper comparisons</h2>
<p>These comparisons appear for the original legacy-CRSP specification. Historical notebook results are a separate rounded benchmark in the comparison CSV.</p>
{''.join(comparisons) if comparisons else '<p>Published-paper comparison does not apply to this sample or specification.</p>'}
<section><h2>Conventions and replication limits</h2><ul>{notes}</ul>
<p>GS uses five annual percentile ranks with weights 5,4,3,2,1. Each May-April holding year starts with equal weights among surviving original constituents; stocks are not re-sorted.
Only months after a stock's final CRSP row within that holding year receive size-decile replacement. Internal missing returns remain missing.</p></section>
<details><summary>Diagnostics</summary>{diagnostics}</details></main></body></html>'''
    html_path = report_dir / "index.html"
    html_path.write_text(markup, encoding="utf-8")
    print(f"Table I report: {html_path}", flush=True)
    if show:
        plt.show()
    return {"html": html_path, "table": report_dir / "table1.csv",
            "comparison": report_dir / "table1_comparison.csv", "average_returns": ar_path, "r1_spreads": spread_path}
