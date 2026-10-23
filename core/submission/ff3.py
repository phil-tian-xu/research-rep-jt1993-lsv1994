"""Assemble Q1-5 from an existing FF3 run; never estimate or download data."""

from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path
from ._html import html_table


def render_ff3_section(result_dir, table1_result_dir, table1_info):
    """Validate the saved regression against the selected original Table I run."""
    result_dir, table1_result_dir = Path(result_dir), Path(table1_result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    if not table1_result_dir.is_absolute():
        table1_result_dir = Path(root_path) / table1_result_dir
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(result_dir / "ff3_coefficients.csv")
    terms = ["alpha", "mkt_rf", "smb", "hml"]
    fields = ["coefficient", "hac_se", "t_stat", "p_value", "ci95_lower", "ci95_upper"]
    if not {"term", *fields}.issubset(frame.columns):
        raise ValueError("Q1-5: missing coefficient or inference columns.")
    if frame["term"].duplicated().any() or set(frame["term"]) != set(terms):
        raise ValueError("Q1-5: require exactly alpha, Mkt-RF, SMB, and HML.")
    if (not np.isfinite(frame[fields].to_numpy()).all() or frame["hac_se"].le(0).any()
            or not frame["p_value"].between(0, 1).all()):
        raise ValueError("Q1-5: invalid estimates, standard errors, or p-values.")
    if (not np.allclose(frame["t_stat"], frame["coefficient"] / frame["hac_se"], rtol=1e-10, atol=1e-12)
            or not (frame["ci95_lower"] <= frame["coefficient"]).all()
            or not (frame["coefficient"] <= frame["ci95_upper"]).all()):
        raise ValueError("Q1-5: inconsistent coefficient inference.")
    universe = table1_info["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"]
    start, end = table1_info["run_start_date"], table1_info["run_end_date"]
    if (info.get("paper") != "jt1993" or info.get("question") != "Q1-5"
            or info.get("universe") != universe or info.get("run_start_date") != start
            or info.get("run_end_date") != end):
        raise ValueError("Q1-5: universe and reporting dates must match the selected original Table I run.")
    saved_source = Path(info["table1_result_dir"])
    if not saved_source.is_absolute():
        saved_source = Path(root_path) / saved_source
    if saved_source.resolve() != table1_result_dir.resolve():
        raise ValueError("Q1-5: the regression was estimated from a different Table I run.")
    returns_file = table1_result_dir / "table1_monthly_returns.csv"
    if info.get("returns_sha256") != sha256(returns_file.read_bytes()).hexdigest():
        raise ValueError("Q1-5: source return checksum changed; rerun the regression before assembly.")
    n_months = len(pd.period_range(start, end, freq="M"))
    if (info.get("n_obs") != n_months or info.get("df_resid") != n_months - 4
            or info.get("panel") not in ("A", "B") or info.get("J") not in (3, 6, 9, 12)
            or info.get("K") not in (3, 6, 9, 12)):
        raise ValueError("Q1-5: incompatible strategy or observation count.")
    if (info.get("return_units") != "decimal monthly returns"
            or info.get("risk_free_subtracted_from_spread") is not False
            or info.get("hac_kernel") != "Bartlett" or info.get("covariance_correction") != "n/(n-4)"
            or "asymptotic normal" not in info.get("inference", "")):
        raise ValueError("Q1-5: incompatible return units, RF treatment, or inference convention.")
    lags = info.get("hac_lags")
    if (not isinstance(lags, int) or isinstance(lags, bool) or not 0 <= lags < n_months
            or not np.isfinite(info.get("r_squared", np.nan)) or not 0 <= info["r_squared"] <= 1):
        raise ValueError("Q1-5: invalid HAC lag count or R-squared.")

    values = frame.set_index("term")
    rows = []
    for term, label in (("alpha", "Alpha (% per month)"), ("mkt_rf", "Mkt-RF"), ("smb", "SMB"), ("hml", "HML")):
        row = values.loc[term]
        scale = 100 if term == "alpha" else 1
        rows.append([label, f"{row['coefficient'] * scale:.4f}", f"{row['hac_se'] * scale:.4f}",
                     f"{row['t_stat']:.3f}", f"{row['p_value']:.3g}",
                     f"{row['ci95_lower'] * scale:.4f}", f"{row['ci95_upper'] * scale:.4f}"])
    table = html_table(
        ["Term", "Estimate", "HAC SE", "HAC t-statistic", "p-value", "95% CI lower", "95% CI upper"],
        rows, "Q1-5. Fama-French three-factor regression",
    )
    alpha = values.loc["alpha"]
    if alpha["coefficient"] > 0 and alpha["p_value"] < 0.05:
        conclusion = ("The momentum strategy retains a positive and statistically significant alpha at the 5% level. "
                      "The three-factor model therefore does not fully explain the spread portfolio’s average return in this sample.")
    elif alpha["coefficient"] > 0:
        conclusion = ("The estimated alpha is positive, although it is not statistically significant at the 5% level. "
                      "This regression does not establish a significant positive risk-adjusted momentum return.")
    else:
        conclusion = "This regression does not provide evidence of a positive and statistically significant momentum alpha."
    sample_label = {"nyse_amex": "NYSE/AMEX", "nyse": "NYSE", "nyse_amex_nasdaq": "NYSE/AMEX/Nasdaq"}[universe]
    window = f"{pd.Timestamp(start):%B %Y}–{pd.Timestamp(end):%B %Y}"
    return f'''<section id="ff3-regression">
  <div class="section-label">Q1-5 / Risk-adjusted momentum returns</div>
  <h2>Does momentum retain a positive FF3 alpha?</h2>
  <p>The ({info['J']},{info['K']}) Panel {escape(info['panel'])} winner-minus-loser strategy is evaluated
    using {sample_label} stocks over {window}, with {n_months} monthly observations.
    Its monthly spread return is regressed on the Fama-French market, size, and value factors:</p>
  <p class="regression-equation">R<sub>W,t</sub> − R<sub>L,t</sub> = α + β<sub>M</sub>(R<sub>M,t</sub> − R<sub>f,t</sub>)
    + s SMB<sub>t</sub> + h HML<sub>t</sub> + ε<sub>t</sub>.</p>
  <p>The dependent variable is the zero-cost spread return. The risk-free rate cancels between
    the two portfolio legs, so it is not subtracted again. Monthly US factors are obtained from the
    <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html"
    target="_blank" rel="noopener noreferrer">Kenneth French Data Library</a> and aligned with the
    holding-return month. All regression inputs are expressed in decimal return units.</p>
  {table}
  <p class="table-note">Alpha, its standard error, and its confidence interval are displayed in percent per month;
    factor loadings are dimensionless. OLS estimates use Newey–West standard errors with {lags} lags,
    Bartlett weights, and n/(n−4) correction. P-values and 95% confidence intervals use two-sided
    asymptotic normal inference. N = {n_months}; R² = {info['r_squared']:.4f}.</p>
  <div class="finding"><p>The estimated monthly alpha is <strong>{alpha['coefficient'] * 100:.3f}%</strong>
    (HAC t = {alpha['t_stat']:.2f}, p = {alpha['p_value']:.3g}), with a 95% confidence interval of
    [{alpha['ci95_lower'] * 100:.3f}%, {alpha['ci95_upper'] * 100:.3f}%]. {conclusion}</p></div>
</section>'''
