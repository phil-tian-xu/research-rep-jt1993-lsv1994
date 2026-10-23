"""Paper-format layouts and saved-output checks for LSV tables."""

import base64
from html import escape
import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.lsv1994.replication._formation import SIGNALS
from core.lsv1994.replication.table1 import STATS
from core.lsv1994.replication.table2 import PAIRS
from core.lsv1994.replication.table4 import MODELS
from core.lsv1994.replication.table5 import METRICS
from core.submission._html import html_table
from utils.path_info import root_path


TABLE6_STRATEGIES = ["C/P tails", "C/P-GS", "B/M tails"]
TERMS = ["intercept", "gs", "bm", "size", "ep_positive", "ep_negative", "cp_positive", "cp_negative"]
TERM_LABELS = ["Int.", "GS", "B/M", "SIZE", "E/P+", "DE/P", "C/P+", "DC/P"]
KEYS = {1: ["strategy", "stat", "decile"], 2: ["strategy", "stat", "group1", "group2"],
        4: ["model", "term"], 5: ["classification", "leg", "section", "metric"], 6: ["strategy", "horizon"]}
VALUES = {1: ["return"], 2: ["return"], 4: ["mean", "t"], 5: ["value"], 6: ["mean", "t"]}


def number(value, decimals=3):
    return f"{value:.{decimals}f}" if pd.notna(value) and np.isfinite(value) else "—"


def validate_summary(frame, table, *, paper=False):
    keys = KEYS[table]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Table {table}: missing or duplicate economic keys.")
    if table == 1:
        expected = {(s,t,d) for s in SIGNALS for t in STATS for d in range(1,11)}
    elif table == 2:
        expected = {(s,t,a,b) for s in PAIRS for t in STATS for a in range(1,4) for b in range(1,4)}
    elif table == 4:
        expected = {(m,t) for m,terms in MODELS.items() for t in ["intercept"]+terms}
    elif table == 5:
        expected = {(s,l,"A" if i<6 else "B" if i<10 else "C",metric)
                    for s in ["B/M","C/P-GS"] for l in ["Glamour","Value"] for i,metric in enumerate(METRICS)}
    else:
        strategies = TABLE6_STRATEGIES if paper else TABLE6_STRATEGIES+["B/M"]
        expected = {(s,h) for s in strategies for h in (1,3,5)}
    if set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"Table {table}: incomplete or unexpected economic keys.")
    for col in VALUES[table]:
        invalid = ~np.isfinite(frame[col].to_numpy(dtype=float))
        if invalid.any():
            allowed = np.zeros(len(frame), dtype=bool)
            if not paper and table == 5 and col == "value":
                allowed = frame["status"].isin(["nonpositive aggregate accounting level", "five-year accounting outcomes unavailable"]).to_numpy()
            if not paper and table == 6 and col == "t":
                allowed = frame["status"].eq("nonpositive HH variance").to_numpy()
            if (invalid & ~allowed).any():
                raise ValueError(f"Table {table}: unexplained nonfinite {col}.")


def load_saved(table, result_dir):
    result_dir = Path(result_dir)
    info = json.loads((result_dir/"run_info.json").read_text(encoding="utf-8"))
    if info.get("paper") != "lsv1994" or info.get("table") != table:
        raise ValueError("Mixed paper/table metadata.")
    for name,digest in info["output_sha256"].items():
        if hashlib.sha256((result_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved output changed: {name}")
    summary = pd.read_csv(result_dir/f"table{table}_summary.csv")
    validate_summary(summary, table)
    cohorts = pd.read_csv(result_dir/"table6_cohort_returns.csv") if table == 6 else None
    return summary, info, cohorts


def table_panels(table, frame, cohorts=None):
    """Return display rows in published panel/model order, independent of HTML."""
    panels = []
    if table in (1,2):
        strategies = SIGNALS if table == 1 else PAIRS
        for strategy in strategies:
            part = frame.loc[frame["strategy"].eq(strategy)]
            columns = list(range(1,11)) if table == 1 else [(a,b) for a in range(1,4) for b in range(1,4)]
            pivot = part.pivot(index="stat", columns="decile" if table==1 else ["group1","group2"], values="return").reindex(index=STATS, columns=columns)
            rows = [[stat]+[number(v,4) for v in pivot.loc[stat]] for stat in STATS]
            headers = ["Statistic"]+[str(c) if table==1 else f"{c[0]},{c[1]}" for c in columns]
            panels.append({"label": f"Table {table} · {strategy}", "headers": headers, "rows": rows})
    elif table == 4:
        indexed = frame.set_index(["model","term"])
        rows=[]
        for model,terms in MODELS.items():
            rows.append([model]+[number(indexed.loc[(model,t),"mean"]) if t in ["intercept"]+terms else "—" for t in TERMS])
            rows.append([""]+[f'({number(indexed.loc[(model,t),"t"],2)})' if t in ["intercept"]+terms else "" for t in TERMS])
        panels.append({"label":"Table IV · Annual cross-sectional regressions", "headers":["Model"]+TERM_LABELS,"rows":rows,"t_rows":True})
    elif table == 5:
        pivot = frame.pivot(index="metric", columns=["classification","leg"],values="value")
        cols=[(s,l) for s in ["B/M","C/P-GS"] for l in ["Glamour","Value"]]
        rows=[[metric]+[number(pivot.loc[metric,c],2 if metric=="SIZE" else 3) for c in cols] for metric in METRICS]
        panels.append({"label":"Table V · Fundamentals and performance", "headers":["Metric","Glamour","Value","Glamour","Value"],
                       "groups":[("",1),("B/M",2),("C/P-GS",2)],"rows":rows})
    else:
        if cohorts is None:
            raise ValueError("Table VI requires saved cohort returns.")
        indexed=cohorts.set_index(["strategy","formation_year","horizon"])["spread"]
        means=frame.set_index(["strategy","horizon"])
        rows=[]
        for year in sorted(cohorts["formation_year"].unique()):
            rows.append([int(year)]+[number(indexed.get((s,year,h),np.nan)) for s in TABLE6_STRATEGIES for h in (1,3,5)])
        for field,label in [("mean","Average"),("t","t-statistic")]:
            rows.append([label]+[number(means.loc[(s,h),field]) for s in TABLE6_STRATEGIES for h in (1,3,5)])
        panels.append({"label":"Table VI · Cohort value-minus-glamour returns", "headers":["Formation"]+["1-year","3-year","5-year"]*3,
                       "groups":[("",1)]+[(s,3) for s in TABLE6_STRATEGIES],"rows":rows})
    return panels


def report_table(table, result_dir):
    frame, info, cohorts = load_saved(table,result_dir)
    directory=Path(result_dir)/"reports"
    directory.mkdir(parents=True,exist_ok=True)
    markup=[]
    if table == 4 and "n_unidentified_years" in frame and frame["n_unidentified_years"].gt(0).any():
        markup.append('<p>Some annual negative-ratio dummies have no observations in the negative category. Those dummy coefficients are unavailable for that year; the other slopes are estimated. Time-series means and t-statistics use the identified years for each term. Annual statuses and per-term year counts are retained in the saved CSVs.</p>')
    if table == 5:
        past_counts = ", ".join(map(str,sorted(frame.loc[frame["section"].ne("C"),"n_cohorts"].unique())))
        future_counts = ", ".join(map(str,sorted(frame.loc[frame["section"].eq("C"),"n_cohorts"].unique())))
        markup.append(f'<p>Current fundamentals and past performance: {past_counts} formation cohorts. Future accounting growth: {future_counts} complete five-year cohorts. Dashes indicate undefined growth from nonpositive aggregate levels or unavailable follow-up; the saved CSV retains each status.</p>')
    for i,panel in enumerate(table_panels(table,frame,cohorts)):
        panel=dict(panel)
        label=panel.pop("label")
        headers=panel.pop("headers")
        rows=panel.pop("rows")
        markup.append(html_table(headers,rows,label,**panel))
        pd.DataFrame(rows,columns=headers).to_csv(directory/f"table{table}_panel{i+1}.csv",index=False)
    if table == 6:
        panel = pd.read_csv(Path(result_dir)/"figure2_data.csv").sort_values("formation_year")
        figure = Figure(figsize=(12,4.8),layout="constrained")
        axis = figure.subplots()
        axis.bar(panel["formation_year"],panel["spread"]*100,color="#16766d")
        low = min(0.,float(panel["spread"].min()*100))-8
        for row in panel.itertuples():
            label = "\n".join(flag for flag,active in [("R",row.recession_overlap),("D",row.market_decline)] if active)
            axis.text(row.formation_year,low,label,ha="center",fontsize=8)
        years = panel["formation_year"].tolist()
        axis.set(xlabel="April formation year",ylabel="One-year value-minus-glamour return (percentage points)",
                 title="Figure 2 · C/P-GS value minus glamour",ylim=(low-3,None),
                 xticks=years[::max(1,int(np.ceil(len(years)/20)))])
        axis.tick_params(axis="x",rotation=60)
        axis.axhline(0,color="#aaa",linewidth=.8)
        image_path = directory/"figure2.png"
        figure.savefig(image_path,dpi=160)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        markup.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="Figure 2: annual C/P-GS value-minus-glamour returns" style="width:100%"><figcaption>R: May-April holding window overlaps an NBER peak-to-trough interval (peak inclusive). D: the CRSP equally weighted index has a negative buy-and-hold return over that window.</figcaption></figure>')
    style=(Path(root_path)/"core/submission/submission.css").read_text(encoding="utf-8")
    method=escape(json.dumps(info["settings"],indent=2))
    plan=escape(json.dumps(info.get("run_plan"),indent=2))
    run_details=f'<details class="methods"><summary>Requested sample and effective coverage</summary><pre>{plan}</pre></details>' if info.get("run_plan") else ""
    caption=escape(f'{info["universe"]} | April formations {info["formation_start"]}-{info["formation_end"]} | holding returns end {info["return_end"]}')
    html=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LSV Table {table}</title><style>{style}</style><main><section><h1>LSV1994 · Table {table}</h1><p>{caption}</p>'+''.join(markup)+run_details+f'<details class="methods"><summary>Construction conventions</summary><pre>{method}</pre></details></section></main></html>'
    html += "<script>document.querySelectorAll('.table-expand').forEach(b=>b.addEventListener('click',()=>{const x=b.closest('.table-panel').classList.toggle('is-expanded');b.setAttribute('aria-expanded',String(x));b.textContent=x?'Collapse table':'Expand table';}));</script>"
    (directory/"index.html").write_text(html,encoding="utf-8")
    frame.to_csv(directory/f"table{table}.csv",index=False)
    return directory/"index.html"
