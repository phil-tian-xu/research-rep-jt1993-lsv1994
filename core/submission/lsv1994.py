"""Assemble Q2-1 through Q2-4 from verified saved results, without raw-data access."""

from html import escape
import hashlib
import json
from pathlib import Path
from string import Template

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.lsv1994.reporter.tables import (
    KEYS, VALUES, TABLE6_STRATEGIES, load_saved, validate_summary, table_panels, number,
)
from core.lsv1994.replication.table2 import PAIRS
from core.lsv1994.replication.table4 import MODELS
from ._html import html_table, figure_html
from utils.path_info import root_path


ROOT=Path(root_path)
PAPER_DIR=ROOT/"references/lsv1994/paper_tables"


def load_results(result_dirs, *, original_sample=True):
    if set(result_dirs) != {1,2,4,5,6}:
        raise ValueError("Q2-1 requires Tables I, II, IV, V, VI and Figure 2.")
    results,info={},{}
    cohorts=None
    for table,directory in result_dirs.items():
        frame,metadata,yearly=load_saved(table,Path(directory))
        results[table],info[table]=frame,metadata
        if table==6:
            cohorts=yearly
    signatures={(m["formation_start"],m["formation_end"],m["return_end"],m["universe"],m["book_equity"],m["accounting_currency"]) for m in info.values()}
    if len(signatures)!=1:
        raise ValueError("LSV tables use different samples, accounting settings, or return cutoffs.")
    start,end,cutoff,universe,_,_=signatures.pop()
    if original_sample and (start,end,cutoff,universe)!=(1968,1989,"1990-04-30","nyse_amex"):
        raise ValueError("Q2-1 paper comparisons require the original NYSE/AMEX formations and April 1990 cutoff.")
    foundations={str(Path(m.get("table1_foundation",result_dirs[1])).resolve()) for m in info.values()}
    if len(foundations)!=1 or foundations!={str(Path(result_dirs[1]).resolve())}:
        raise ValueError("LSV tables were not built from the same Table I foundation.")
    raw_paths={m["raw_inputs"]["crsp_monthly"]["path"] for m in info.values()}
    if len(raw_paths)!=1 or len({json.dumps(m["raw_inputs"],sort_keys=True) for m in info.values()})!=1:
        raise ValueError("LSV tables use different raw-data sources or data vintages.")
    cutoff_date=pd.Timestamp(cutoff)
    if cutoff_date.month!=4 or cutoff_date.day!=30 or end>cutoff_date.year-1:
        raise ValueError("Submission cases require complete May-April holding years.")
    if cohorts.duplicated(["strategy","formation_year","horizon"]).any():
        raise ValueError("Duplicate Table VI cohort keys.")
    for horizon in (1,3,5):
        years=set(range(start,min(end,cutoff_date.year-horizon)+1))
        if not results[6].loc[results[6]["horizon"].eq(horizon),"n_cohorts"].eq(len(years)).all():
            raise ValueError("Table VI cohort counts disagree with holding cutoffs.")
        for strategy in results[6]["strategy"].unique():
            actual=cohorts.loc[cohorts["strategy"].eq(strategy)&cohorts["horizon"].eq(horizon),"formation_year"]
            if set(actual)!=years:
                raise ValueError("Table VI is missing a complete calendar cohort.")
    figure_data=pd.read_csv(Path(result_dirs[6])/"figure2_data.csv")
    expected=cohorts.loc[cohorts["strategy"].eq("C/P-GS") & cohorts["horizon"].eq(1)].sort_values("formation_year")
    figure_data=figure_data.sort_values("formation_year")
    if (figure_data["formation_year"].tolist()!=expected["formation_year"].tolist()
            or not np.allclose(figure_data["spread"],expected["spread"],atol=1e-12,rtol=1e-12)):
        raise ValueError("Figure 2 disagrees with Table VI's annual C/P-GS series.")
    return results,info,cohorts,figure_data


def render_results(table,frame,cohorts=None,*,label_prefix="My results"):
    parts=[]
    for panel in table_panels(table,frame,cohorts):
        panel=dict(panel)
        label,headers,rows=panel.pop("label"),panel.pop("headers"),panel.pop("rows")
        parts.append(html_table(headers,rows,label_prefix+" · "+label,**panel))
    return ''.join(parts)


def render_comparison(table,frame,paper,cohorts,paper_cohorts):
    keys=KEYS[table]
    compare=frame.merge(paper,on=keys,suffixes=("_my","_paper"),validate="one_to_one",how="inner")
    if len(compare)!=len(paper):
        raise ValueError(f"Table {table}: incomplete paper benchmark join.")
    parts=[]
    if table in (1,2):
        for strategy in paper["strategy"].unique():
            cell=compare.loc[compare["strategy"].eq(strategy)]
            columns=list(range(1,11)) if table==1 else [(a,b) for a in range(1,4) for b in range(1,4)]
            index="decile" if table==1 else ["group1","group2"]
            my=cell.pivot(index="stat",columns=index,values="return_my").reindex(columns=columns)
            original=cell.pivot(index="stat",columns=index,values="return_paper").reindex(columns=columns)
            stats=["R1","R2","R3","R4","R5","AR","CR5","SAAR"]
            rows=[[stat]+[number(v,4) for v in my.loc[stat]]+[number(v) for v in original.loc[stat]] for stat in stats]
            labels=[str(c) if table==1 else f"{c[0]},{c[1]}" for c in columns]
            parts.append(html_table(["Statistic"]+labels*2,rows,f"Table {table} · {strategy} · Replication versus paper",
                                    groups=[("",1),("My results",len(columns)),("Published paper",len(columns))]))
    elif table==4:
        rows=[[int(r.model),r.term,number(r.mean_my),number(r.t_my),number(r.mean_paper),number(r.t_paper),number(r.mean_my-r.mean_paper)] for r in compare.itertuples()]
        parts.append(html_table(["Model","Term","Coefficient","t","Coefficient","t","Difference"],rows,
                                "Table IV · All coefficient and t-statistic comparisons",groups=[("",2),("My results",2),("Published paper",2),("My − paper",1)]))
    elif table==5:
        columns=[(s,l) for s in ["B/M","C/P-GS"] for l in ["Glamour","Value"]]
        my=frame.pivot(index="metric",columns=["classification","leg"],values="value")
        original=paper.pivot(index="metric",columns=["classification","leg"],values="value")
        from core.lsv1994.replication.table5 import METRICS
        rows=[[metric]+[number(my.loc[metric,c],2 if metric=="SIZE" else 3) for c in columns]
              +[number(original.loc[metric,c],2 if metric=="SIZE" else 3) for c in columns] for metric in METRICS]
        parts.append(html_table(["Metric"]+["B/M G","B/M V","C/P-GS G","C/P-GS V"]*2,rows,
                                "Table V · Replication versus paper",groups=[("",1),("My results",4),("Published paper",4)]))
    else:
        for strategy in TABLE6_STRATEGIES:
            mine=cohorts.loc[cohorts["strategy"].eq(strategy)].pivot(index="formation_year",columns="horizon",values="spread")
            original=paper_cohorts.loc[paper_cohorts["strategy"].eq(strategy)].pivot(index="formation_year",columns="horizon",values="spread")
            rows=[[year]+[number(mine.loc[year].get(h,np.nan)) for h in (1,3,5)]
                  +[number(original.loc[year].get(h,np.nan)) for h in (1,3,5)] for year in range(1968,1990)]
            cell=compare.loc[compare["strategy"].eq(strategy)].set_index("horizon")
            for field,label in [("mean","Average"),("t","t-statistic")]:
                rows.append([label]+[number(cell.loc[h,field+"_my"]) for h in (1,3,5)]
                            +[number(cell.loc[h,field+"_paper"]) for h in (1,3,5)])
            parts.append(html_table(["Formation"]+["1-year","3-year","5-year"]*2,rows,
                                    f"Table VI · {strategy} · Replication versus paper",groups=[("",1),("My results",3),("Published paper",3)]))
    return ''.join(parts)


def return_figures(results,paper):
    figures={}
    figure=Figure(figsize=(11,7),layout="constrained")
    for ax,strategy in zip(figure.subplots(2,2).flat,["B/M","C/P","E/P","GS"]):
        for frame,label,style in [(results[1],"My results","o-"),(paper[1],"Paper","s--")]:
            x=frame.loc[frame["strategy"].eq(strategy)&frame["stat"].eq("AR")].sort_values("decile")
            ax.plot(x["decile"],x["return"]*100,style,label=label,markersize=4)
        ax.set(title=strategy,xlabel="Ascending decile",ylabel="Average annual return (%)",xticks=range(1,11))
        ax.legend(frameon=False,fontsize=9)
    figures[1]=figure_html(figure,"Average annual returns across all four one-dimensional sorts. GS runs from value to glamour; the other signals run from glamour to value.")
    figure=Figure(figsize=(11,4.5),layout="constrained")
    for ax,stat in zip(figure.subplots(1,2),["AR","SAAR"]):
        for offset,frame,label in [(-.18,results[2],"My results"),(.18,paper[2],"Paper")]:
            values=[]
            for strategy,signals in PAIRS.items():
                cell=frame.loc[frame["strategy"].eq(strategy)&frame["stat"].eq(stat)].set_index(["group1","group2"])["return"]
                glamour,value=((1,3),(3,1)) if signals[1]=="gs" else ((1,1),(3,3))
                values.append((cell[value]-cell[glamour])*100)
            ax.bar(np.arange(5)+offset,values,.35,label=label)
        ax.set(title=stat+": value minus glamour",ylabel="Percentage points per year")
        ax.set_xticks(range(5),list(PAIRS),rotation=20)
        ax.axhline(0,color="#999",linewidth=.7)
        ax.legend(frameon=False)
    figures[2]=figure_html(figure,"Extreme-portfolio raw and size-adjusted annual spreads for each independent two-dimensional sort.")
    figure=Figure(figsize=(12,3.6),layout="constrained")
    for ax,model,term in zip(figure.subplots(1,5),range(1,6),["gs","bm","size","ep_positive","cp_positive"]):
        for pos,frame,label,color in [(0,results[4],"My results","#16766d"),(1,paper[4],"Paper","#687180")]:
            row=frame.loc[frame["model"].eq(model)&frame["term"].eq(term)].iloc[0]
            se=row["se"] if "se" in row else abs(row["mean"]/row["t"])
            ax.errorbar(pos,row["mean"],yerr=1.96*se,fmt="o",capsize=4,color=color)
        ax.axhline(0,color="#aaa",linewidth=.8)
        ax.set(title=term.replace("_positive","+"),xticks=[0,1],xticklabels=["Mine","Paper"])
    figures[4]=figure_html(figure,"Single-characteristic regression slopes, with 1.96 standard-error bands based on the time series of annual coefficients. Slopes have different units across panels.")
    return figures


def growth_figure(results,paper):
    figure=Figure(figsize=(12,4.2),layout="constrained")
    for ax,interval,title in zip(figure.subplots(1,3),["-5,0","0,5","2,5"],["Past five years","Future five years","Future years 2–5"]):
        for offset,frame,leg,label in [(-.2,results,"Glamour","My glamour"),(.2,results,"Value","My value")]:
            values=[frame.loc[frame["classification"].eq("C/P-GS")&frame["leg"].eq(leg)&frame["metric"].eq(prefix+"("+interval+")"),"value"].iloc[0]*100 for prefix in ["AEG","ACG","ASG"]]
            ax.bar(np.arange(3)+offset,values,.38,label=label)
            original=[paper.loc[paper["classification"].eq("C/P-GS")&paper["leg"].eq(leg)&paper["metric"].eq(prefix+"("+interval+")"),"value"].iloc[0]*100 for prefix in ["AEG","ACG","ASG"]]
            ax.scatter(np.arange(3)+offset,original,marker="_",s=120,color="black")
        ax.set(title=title,ylabel="Annual growth (%)",xticks=range(3),xticklabels=["Earnings","Cash flow","Sales"])
        ax.axhline(0,color="#aaa",linewidth=.8)
        ax.legend(frameon=False,fontsize=8)
    return figure_html(figure,"C/P-GS portfolio accounting growth. Bars are calculated results; black marks are published values. Undefined growth from nonpositive aggregate levels is left blank.")


def risk_figure(figure_data,paper=None,*,label="C/P-GS value minus glamour"):
    figure=Figure(figsize=(12,4.8),layout="constrained")
    ax=figure.subplots()
    panel=figure_data.sort_values("formation_year")
    ax.bar(panel["formation_year"],panel["spread"]*100,color="#16766d",label="My results")
    low=min(0.,float(panel["spread"].min()*100))-8
    if paper is not None:
        original=paper.loc[paper["strategy"].eq("C/P-GS")&paper["horizon"].eq(1)].sort_values("formation_year")
        ax.plot(original["formation_year"],original["spread"]*100,"o--",color="#687180",label="Published paper",markersize=3)
        low=min(low,float(original["spread"].min()*100)-8)
    for row in panel.itertuples():
        flags="\n".join(flag for flag,active in [("R",row.recession_overlap),("D",row.market_decline)] if active)
        ax.text(row.formation_year,low,flags,ha="center",fontsize=8)
    years=panel["formation_year"].tolist()
    step=1 if len(years)<=26 else int(np.ceil(len(years)/20))
    ax.set(xlabel="April formation year",ylabel="One-year value-minus-glamour return (percentage points)",
           title="Figure 2 · "+label,ylim=(low-3,None),xticks=years[::step])
    ax.tick_params(axis="x",rotation=60)
    ax.axhline(0,color="#aaa",linewidth=.8)
    if paper is not None:
        ax.legend(frameon=False)
    return figure_html(figure,"R denotes a May-April holding window overlapping the versioned NBER peak-to-trough dates, including the peak month; D denotes a negative CRSP equally weighted index buy-and-hold return over that window. The R convention may differ from printed marker placement.")


EXTENSION_CASES = {
    "nyse": ("Q2-2", "NYSE", "nyse", "1962-01-01", "2025-12-31"),
    "nyse_amex_nasdaq": ("Q2-2", "NYSE/AMEX/Nasdaq", "nyse_amex_nasdaq", "1962-01-01", "2025-12-31"),
    "early": ("Q2-3", "1963–1989", "nyse_amex", "1963-01-01", "1989-12-31"),
    "recent": ("Q2-3", "1990–2025", "nyse_amex", "1990-01-01", "2025-12-31"),
}


def load_extended_cases(case_dirs):
    if set(case_dirs)!=set(EXTENSION_CASES):
        raise ValueError("Extended submission requires both Q2-2 universes and both Q2-3 subsamples.")
    cases={}
    for key,(question,label,universe,start,end) in EXTENSION_CASES.items():
        case=load_results(case_dirs[key],original_sample=False)
        results,info,_,_=case
        meta=info[1]
        plan=meta.get("run_plan") or {}
        if ((plan.get("question"),meta["universe"],plan.get("requested_sample_start"),plan.get("requested_sample_end"))
                !=(question,universe,start,end)):
            raise ValueError(f"{key}: saved run does not match the requested assignment case.")
        if plan.get("universe")!=meta["universe"]:
            raise ValueError(f"{key}: requested and effective universes disagree.")
        for table,table_info in info.items():
            if table_info.get("run_plan")!=plan:
                raise ValueError(f"{key}: tables have inconsistent requested/effective coverage plans.")
        if (meta["formation_start"]!=plan["formation_start"] or meta["formation_end"]!=plan["formation_end"]
                or meta["return_end"]!=plan["return_end"] or meta["formation_start"]<pd.Timestamp(start).year
                or pd.Timestamp(meta["return_end"])>pd.Timestamp(end) or plan["outcome_end_year"]>pd.Timestamp(end).year):
            raise ValueError(f"{key}: returns or accounting outcomes exceed the requested sample window.")
        outcome_year=info[5].get("settings",{}).get("outcome_end_year")
        if outcome_year!=plan["outcome_end_year"]:
            raise ValueError(f"{key}: Table V uses a different accounting-outcome endpoint.")
        n=meta["formation_end"]-meta["formation_start"]+1
        future=max(0,min(meta["formation_end"],outcome_year-4)-meta["formation_start"]+1)
        for section,count in [("A",n),("B",n),("C",future)]:
            if not results[5].loc[results[5]["section"].eq(section),"n_cohorts"].eq(count).all():
                raise ValueError(f"{key}: Table V follow-up cohort counts disagree with the sample window.")
        cases[key]=case
    sources={json.dumps(case[1][1]["raw_inputs"],sort_keys=True) for case in cases.values()}
    settings={(case[1][1]["book_equity"],case[1][1]["accounting_currency"]) for case in cases.values()}
    if len(sources)!=1 or len(settings)!=1:
        raise ValueError("Extended comparisons require the same raw-data vintage and accounting settings.")
    signatures={(cases[key][1][1]["formation_start"],cases[key][1][1]["formation_end"],cases[key][1][1]["return_end"])
                for key in ("nyse","nyse_amex_nasdaq")}
    if len(signatures)!=1:
        raise ValueError("Q2-2 universes must use matching formation and holding windows.")
    return cases


def coverage_overview(cases):
    rows=[["Q2-1", "Original paper", "NYSE/AMEX", "1968–1989", "May 1968–April 1990", "22 / 21 / 20 / 19 / 18", "22"]]
    for key,case in cases.items():
        question,label,universe,requested_start,requested_end=EXTENSION_CASES[key]
        meta=case[1][1]
        plan=meta["run_plan"]
        n=meta["formation_end"]-meta["formation_start"]+1
        future=int(case[0][5].loc[case[0][5]["section"].eq("C"),"n_cohorts"].iloc[0])
        market={"nyse":"NYSE","nyse_amex":"NYSE/AMEX","nyse_amex_nasdaq":"NYSE/AMEX/Nasdaq"}[universe]
        rows.append([question,requested_start[:4]+"–"+requested_end[:4],market,
                     f'{meta["formation_start"]}–{meta["formation_end"]}',
                     f'May {meta["formation_start"]}–April {pd.Timestamp(meta["return_end"]).year}',
                     " / ".join(str(n-h+1) for h in range(1,6)),str(future)])
    table=html_table(["Task","Requested sample","Universe","April formations","Holding returns","R1 / R2 / R3 / R4 / R5 cohorts","Future-growth cohorts"],rows,"Assignment coverage and effective samples")
    return ('<section id="coverage"><div class="section-label">Assignment coverage</div><h2>Original replication and available-data extensions</h2>'
            '<aside class="convention"><strong>The requested extended period is not fully covered.</strong> The shared cache starts in 1957 and ends in December 2024. '
            'Five-year GS signals first support April 1963 formations; April 1962 needs accounting history from 1956. Complete May–April holding years end in April 2024. '
            'Thus Q2-2 and the recent Q2-3 sample below do not include 2025 returns. These are the available-data parts of the assignment.</aside>'+table+
            '<p class="table-note">Cohort counts refer to complete annual holding horizons. For Q2-3, the early calendar window stops in 1989 and the recent window starts in 1990. '
            'The annual definitions omit the 1989 formation, whose first holding year crosses the boundary, and do not import portfolios formed before the recent window. '
            'Warm-up history may precede a window; holding returns and future accounting growth cannot exceed its endpoint.</p></section>')


def comparison_tables(cases,section_id):
    entries=list(cases.items())
    labels=[EXTENSION_CASES[key][1] for key,_ in entries]
    rows=[]
    for strategy in ("B/M","C/P","E/P","GS"):
        for stat in ("R1","AR","SAAR"):
            values=[]
            for _,case in entries:
                cell=case[0][1].loc[case[0][1]["strategy"].eq(strategy)&case[0][1]["stat"].eq(stat)].set_index("decile")["return"]
                values.append(float(cell[1]-cell[10] if strategy=="GS" else cell[10]-cell[1])*100)
            rows.append([strategy,stat]+[number(value,2) for value in values]+[number(values[1]-values[0],2)])
    spread=html_table(["Signal","Statistic"]+labels+[labels[1]+" − "+labels[0]],rows,section_id+" · Table I value-minus-glamour comparison")
    rows=[]
    for strategy in TABLE6_STRATEGIES+["B/M"]:
        for horizon in (1,3,5):
            display=strategy if strategy!="B/M" else "B/M 10−1 (sensitivity)"
            values=[]
            means=[]
            for _,case in entries:
                row=case[0][6].set_index(["strategy","horizon"]).loc[(strategy,horizon)]
                means.append(float(row["mean"])*100)
                values.extend([number(row["mean"]*100,2),number(row["t"],2),str(int(row["n_cohorts"]))])
            rows.append([display,str(horizon)]+values+[number(means[1]-means[0],2)])
    risk=html_table(["Strategy","Years"]+["Mean (pp)","t","N"]*2+["Difference (pp)"],rows,section_id+" · Table VI premium and inference comparison",
                    groups=[("",2),(labels[0],3),(labels[1],3),(labels[1]+" − "+labels[0],1)])
    return ('<p class="table-note">Comparison spreads and differences are in percentage points. AR averages the five annual holding-horizon means; SAAR averages the corresponding size-adjusted means. Table VI three- and five-year spreads are cumulative rather than annualized. N counts complete formation cohorts.</p>'+spread+risk)


def comparison_figures(cases):
    entries=list(cases.items())
    output=[]
    figure=Figure(figsize=(11,7),layout="constrained")
    for axis,strategy in zip(figure.subplots(2,2).flat,("B/M","C/P","E/P","GS")):
        for (key,case),style in zip(entries,("o-","s--")):
            frame=case[0][1]
            cell=frame.loc[frame["strategy"].eq(strategy)&frame["stat"].eq("AR")].sort_values("decile")
            axis.plot(cell["decile"],cell["return"]*100,style,label=EXTENSION_CASES[key][1],markersize=4)
        axis.set(title=strategy,xlabel="Ascending decile",ylabel="Average annual return (%)",xticks=range(1,11))
        axis.legend(frameon=False,fontsize=8)
    output.append(figure_html(figure,"Table I average annual decile returns, using each case's complete horizons. GS runs from value to glamour; valuation-ratio deciles run from glamour to value."))
    figure=Figure(figsize=(11,4.5),layout="constrained")
    for axis,stat in zip(figure.subplots(1,2),("AR","SAAR")):
        for (key,case),offset in zip(entries,(-.18,.18)):
            values=[]
            frame=case[0][2]
            for strategy,signals in PAIRS.items():
                cell=frame.loc[frame["strategy"].eq(strategy)&frame["stat"].eq(stat)].set_index(["group1","group2"])["return"]
                glamour,value=((1,3),(3,1)) if signals[1]=="gs" else ((1,1),(3,3))
                values.append((cell[value]-cell[glamour])*100)
            axis.bar(np.arange(5)+offset,values,.35,label=EXTENSION_CASES[key][1])
        axis.set(title=stat+": value minus glamour",ylabel="Percentage points per year")
        axis.set_xticks(range(5),list(PAIRS),rotation=20)
        axis.axhline(0,color="#aaa",linewidth=.8)
        axis.legend(frameon=False,fontsize=8)
    output.append(figure_html(figure,"Table II extreme-portfolio annual raw and size-adjusted spreads for all five independent double sorts."))
    figure=Figure(figsize=(12,4.2),layout="constrained")
    for axis,strategy in zip(figure.subplots(1,3),TABLE6_STRATEGIES):
        for (key,case),offset in zip(entries,(-.18,.18)):
            cell=case[0][6].loc[case[0][6]["strategy"].eq(strategy)].set_index("horizon").reindex([1,3,5])
            axis.bar(np.arange(3)+offset,cell["mean"]*100,.35,yerr=1.96*cell["se"]*100,capsize=3,label=EXTENSION_CASES[key][1])
        axis.set(title=strategy,ylabel="Cumulative spread (percentage points)",xticks=range(3),xticklabels=["1-year","3-year","5-year"])
        axis.axhline(0,color="#aaa",linewidth=.8)
        axis.legend(frameon=False,fontsize=8)
    output.append(figure_html(figure,"Table VI average cohort spreads with 1.96 Hansen–Hodrick standard-error bands. Three- and five-year returns are cumulative, not annualized. These bands describe each mean; they do not test the difference between samples."))
    return ''.join(output)


def complete_cases(cases,section_id):
    options,panels=[],[]
    for index,(key,case) in enumerate(cases.items()):
        results,info,cohorts,figure=case
        question,label,_,_,_=EXTENSION_CASES[key]
        case_id=section_id+"-"+key
        options.append(f'<option value="{case_id}">{escape(label)}</option>')
        hidden=" hidden" if index else ""
        prefix=question+" · "+label
        meta=info[1]
        plan=meta["run_plan"]
        future=results[5].loc[results[5]["section"].eq("C"),"n_cohorts"].iloc[0]
        undefined=int(results[5]["value"].isna().sum())
        currencies=meta["diagnostics"]["formation_account_currencies"]
        nonusd=sum(n for currency,n in currencies.items() if currency not in ("USD","missing"))
        notes=(f'April formations {meta["formation_start"]}–{meta["formation_end"]}; holding returns end {meta["return_end"]}. '
               f'Table V future growth uses {int(future)} complete five-year follow-up cohorts, with outcomes through {plan["outcome_end_year"]}; '
               f'{undefined} growth cells are undefined because aggregate accounting levels are nonpositive. '
               f'{nonusd:,} formation observations have non-USD accounts without currency conversion.')
        if "n_unidentified_years" in results[4]:
            omitted=results[4].loc[results[4]["n_unidentified_years"].gt(0)]
            if len(omitted):
                terms="; ".join(f'model {int(row.model)} {row.term}: {int(row.n_years)}/{int(row.n_total_years)} years' for row in omitted.itertuples())
                notes+=' Table IV annual negative-ratio dummies with no negative observations remain unavailable: '+terms+'. Other slopes retain their identified annual estimates.'
        tables=''.join(render_results(table,results[table],cohorts,label_prefix=prefix) for table in (1,2,4,5,6))
        panels.append(f'<article class="case-panel" id="{case_id}"{hidden}><h4>{escape(label)}</h4><p class="table-note">{escape(notes)}</p>'
                      '<p class="table-note">The complete tables use decimal returns and growth rates; 0.1000 means 10%. SIZE is USD millions. '
                      'The comparison summaries above use percentage points. Expand any table to display all rows.</p>'+tables+
                      '<h3>Figure 2 · Annual C/P–GS performance</h3>'+risk_figure(figure,label=label+f' · {meta["formation_start"]}–{meta["formation_end"]} formations')+'</article>')
    return (f'<details class="full-results" open><summary>Complete Tables I, II, IV, V, VI and Figure 2 for each case</summary>'
            f'<div class="case-picker"><label for="{section_id}-select">Show results for</label>'
            f'<select class="case-select" id="{section_id}-select" data-group="{section_id}-cases">'+''.join(options)+
            f'</select></div><div id="{section_id}-cases">'+''.join(panels)+'</div></details>')


def extension_sections(cases):
    universes={key:cases[key] for key in ("nyse","nyse_amex_nasdaq")}
    periods={key:cases[key] for key in ("early","recent")}
    q22=('<section id="exchange-samples"><div class="section-label">Q2-2 / Exchange universes</div><h2>NYSE and NYSE/AMEX/Nasdaq</h2>'
         '<p>Both universes use the same shared CRSP/Compustat/CCM vintage, April formation dates, accounting definitions and holding-return endpoints. '
         'Ranks and size benchmarks are recomputed for each selected stock universe. The broader universe adds AMEX and Nasdaq as their historical exchange codes become available.</p>'
         '<p>Table I spreads subtract glamour from value (decile 10 minus 1 for valuation ratios; 1 minus 10 for GS). Table VI uses pooled 9/10 minus 1/2 tails for C/P and B/M, '
         'so those spreads are different portfolio definitions. The B/M 10−1 sensitivity is labelled separately. All differences below are descriptive comparisons between overlapping stock universes.</p>'+
         comparison_tables(universes,"Q2-2")+comparison_figures(universes)+complete_cases(universes,"exchange-samples")+'</section>')
    annual={key:case[0][6].loc[case[0][6]["horizon"].eq(1)].set_index("strategy") for key,case in periods.items()}
    early,recent=annual["early"],annual["recent"]
    lower=sum(recent.loc[strategy,"mean"]<early.loc[strategy,"mean"] for strategy in TABLE6_STRATEGIES)
    lower_t=sum(abs(recent.loc[strategy,"t"])<abs(early.loc[strategy,"t"]) for strategy in TABLE6_STRATEGIES)
    inference=("The recent absolute t-statistics are lower for all three main annual strategies. " if lower_t==3
               else "Changes in absolute t-statistics vary across the three main annual strategies. ")
    conclusion=("All three Table VI average annual premiums are lower in the recent sample." if lower==3 else
                "None of the three Table VI average annual premiums is lower in the recent sample." if lower==0 else
                "The change in average annual value premiums is mixed across the three Table VI definitions.")
    finding=(f'For C/P–GS, the average one-year premium changes from {early.loc["C/P-GS","mean"]*100:.2f}% '
             f'(t = {early.loc["C/P-GS","t"]:.2f}) to {recent.loc["C/P-GS","mean"]*100:.2f}% '
             f'(t = {recent.loc["C/P-GS","t"]:.2f}). The C/P tails premium changes from {early.loc["C/P tails","mean"]*100:.2f}% '
             f'to {recent.loc["C/P tails","mean"]*100:.2f}%, while B/M tails change from {early.loc["B/M tails","mean"]*100:.2f}% '
             f'to {recent.loc["B/M tails","mean"]*100:.2f}%. {conclusion}')
    ar_changes=[]
    lower_ar,lower_saar=0,0
    for strategy in ("B/M","C/P","E/P","GS"):
        premia={}
        for stat in ("AR","SAAR"):
            values=[]
            for case in periods.values():
                frame=case[0][1]
                cell=frame.loc[frame["strategy"].eq(strategy)&frame["stat"].eq(stat)].set_index("decile")["return"]
                values.append(float(cell[1]-cell[10] if strategy=="GS" else cell[10]-cell[1])*100)
            premia[stat]=values
        lower_ar+=premia["AR"][1]<premia["AR"][0]
        lower_saar+=premia["SAAR"][1]<premia["SAAR"][0]
        ar_changes.append(f'{strategy}: {premia["AR"][0]:.2f} to {premia["AR"][1]:.2f}')
    horizon_finding=(f'Table I AR, averaging annual holding years 1–5, has lower decile premiums for {lower_ar} of four signals '
                     f'({"; ".join(ar_changes)} percentage points). Size-adjusted SAAR premiums are lower for {lower_saar} of four signals. '
                     'The evidence of weakening therefore depends on the return measure and portfolio definition; positive recent premiums remain visible in the tables.')
    q23=('<section id="subperiods"><div class="section-label">Q2-3 / Subsample comparison</div><h2>Does the value effect weaken in the recent period?</h2>'
         '<p>The requested split is 1963–1989 versus 1990–2025. Both calculations use NYSE/AMEX common stocks, retaining the original paper universe; '
         'the actual formation and return dates are listed in the coverage table. Each subsample recomputes all five tables and Figure 2 using complete horizons entirely within its calendar window.</p>'
         '<div class="finding"><p>'+escape(finding)+'</p><p>'+escape(horizon_finding)+'</p><p>These estimates do not establish a uniform disappearance of the value effect. '
         +inference+'A lower t-statistic can reflect higher return uncertainty; '
         'it is not a test that the average premium fell. No formal test of a structural break or difference in means is reported here, and the missing 2025 endpoint limits the assignment comparison.</p></div>'+
         comparison_tables(periods,"Q2-3")+comparison_figures(periods)+complete_cases(periods,"subperiods")+'</section>')
    return q22+q23


def generate_submission(result_dirs,*,paper_dir=PAPER_DIR,output_path=None,extension_dirs=None):
    results,info,cohorts,figure_data=load_results(result_dirs)
    cases=load_extended_cases(extension_dirs) if extension_dirs is not None else {}
    if cases and any((case[1][1]["book_equity"],case[1][1]["accounting_currency"])!=
                     (info[1]["book_equity"],info[1]["accounting_currency"]) for case in cases.values()):
        raise ValueError("Original and extended sections must use consistent accounting settings.")
    paper_dir=Path(paper_dir)
    paper={table:pd.read_csv(paper_dir/f"table{table}.csv") for table in (1,2,4,5,6)}
    for table,frame in paper.items():
        validate_summary(frame,table,paper=True)
    paper_cohorts=pd.read_csv(paper_dir/"table6_cohorts.csv")
    if len(paper_cohorts)!=180 or paper_cohorts.duplicated(["strategy","formation_year","horizon"]).any():
        raise ValueError("Incomplete Table VI published cohort benchmark.")
    figure=return_figures(results,paper)
    figure[5]=growth_figure(results[5],paper[5])
    figure[6]=risk_figure(figure_data,paper_cohorts)
    substitutions={"style":(ROOT/"core/submission/submission.css").read_text(encoding="utf-8"),
                   "report_scope":"Question 2" if cases else "Question 2-1",
                   "coverage_overview":coverage_overview(cases) if cases else "",
                   "extension_sections":extension_sections(cases) if cases else "",
                   "extension_nav":'<a href="#exchange-samples">Q2-2 · Universes</a><a href="#subperiods">Q2-3 · Subsamples</a>' if cases else "",
                   "extension_badge":"<span>Q2-2 / Q2-3: available-data extensions</span>" if cases else ""}
    for table in results:
        substitutions[f"table{table}"]=render_results(table,results[table],cohorts)
        substitutions[f"comparison{table}"]=render_comparison(table,results[table],paper[table],cohorts,paper_cohorts)
        substitutions[f"figure{table}"]=figure[table]
    cp=results[2].loc[results[2]["strategy"].eq("C/P-GS")&results[2]["stat"].eq("AR")].set_index(["group1","group2"])["return"]
    substitutions["cp_ar_spread"]=number((cp[3,1]-cp[1,3])*100,2)
    substitutions["undefined_growth_count"]=str(int(results[5]["value"].isna().sum()))
    bm_glamour=results[5].loc[results[5]["classification"].eq("B/M") & results[5]["leg"].eq("Glamour") & results[5]["metric"].eq("B/M"),"value"].iloc[0]
    substitutions["bm_glamour_ratio"]=number(bm_glamour)
    annual=figure_data["spread"]
    five=cohorts.loc[cohorts["strategy"].eq("C/P-GS")&cohorts["horizon"].eq(5),"spread"]
    substitutions.update({"annual_wins":str(int(annual.gt(0).sum())),"five_wins":str(int(five.gt(0).sum())),
                          "five_total":str(len(five)),"worst_annual":number(annual.min()*100,2),
                          "currency_count":str(sum(n for c,n in info[1]["diagnostics"]["formation_account_currencies"].items() if c not in ("USD","missing"))),
                          "currency_setting":escape(str(info[1]["accounting_currency"] or "reference convention; no currency restriction"))})
    template=Template((ROOT/"core/submission/lsv1994_template.html").read_text(encoding="utf-8"))
    html=template.substitute(substitutions)
    output_path=Path(output_path) if output_path is not None else ROOT/"results/lsv1994/submission/submission.html"
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(html,encoding="utf-8")
    metadata={"q21":info,**{key:case[1] for key,case in cases.items()}}
    manifest={"paper":"lsv1994","sections":["Q2-1","Q2-2","Q2-3","Q2-4"] if cases else ["Q2-1","Q2-4"],
              "saved_results_only":True,"submission_sha256":hashlib.sha256(output_path.read_bytes()).hexdigest(),
              "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "template_sha256":hashlib.sha256((ROOT/"core/submission/lsv1994_template.html").read_bytes()).hexdigest(),
              "cases":{key:{"sample":{field:tables[1][field] for field in
                         ("formation_start","formation_end","return_end","universe","book_equity","accounting_currency")},
                         "run_plan":tables[1].get("run_plan"),
                         "input_output_sha256":{table:meta["output_sha256"] for table,meta in tables.items()}}
                       for key,tables in metadata.items()}}
    output_path.with_name(output_path.stem+"_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return output_path
