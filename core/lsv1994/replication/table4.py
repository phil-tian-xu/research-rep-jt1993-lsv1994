"""Table IV: nine annual cross-sectional specifications and Fama-MacBeth inference."""

import numpy as np
import pandas as pd


MODELS = {
    1: ["gs"], 2: ["bm"], 3: ["size"], 4: ["ep_positive"], 5: ["cp_positive"],
    6: ["gs", "bm", "cp_positive", "cp_negative"],
    7: ["bm", "size", "cp_positive", "cp_negative"],
    8: ["gs", "bm", "size", "cp_positive", "cp_negative"],
    9: ["gs", "bm", "size", "ep_positive", "ep_negative"],
}


def build_table4(formation, annual_stock):
    data = formation.copy()
    data["size"] = np.log(data["me_apr"])
    for signal in ("ep", "cp"):
        data[f"{signal}_positive"] = data[signal].clip(lower=0)
        # Preserve missing ratios as missing, rather than encoding them as zero.
        data[f"{signal}_negative"] = data[signal].lt(0).astype(float).where(data[signal].notna())
    data = data.merge(annual_stock[["permno", "start_year", "annual_ret"]].rename(
        columns={"start_year": "formation_year"}), on=["permno", "formation_year"],
        how="left", validate="one_to_one")
    coefficients = []
    for year, cohort in data.groupby("formation_year"):
        for model, terms in MODELS.items():
            sample = cohort[["annual_ret"] + terms].replace([np.inf, -np.inf], np.nan).dropna()
            y = sample["annual_ret"].to_numpy()
            # With no negative ratios, the dummy is identically zero. Other
            # slopes remain identified; its annual coefficient is unavailable.
            omitted = {term for term in terms if term.endswith("_negative")
                       and len(sample) and sample[term].eq(0).all()}
            active = [term for term in terms if term not in omitted]
            x = np.column_stack([np.ones(len(sample)), sample[active].to_numpy()])
            if len(sample) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
                raise ValueError(f"Table IV model {model}, {year}: insufficient observations or unidentified regressors.")
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            residual = y - x @ beta
            total = np.square(y-y.mean()).sum()
            estimates = dict(zip(["intercept"] + active, beta))
            for term in ["intercept"] + terms:
                value = estimates.get(term, np.nan)
                coefficients.append({"formation_year": year, "model": model, "term": term,
                                     "coefficient": value, "n_stocks": len(y),
                                     "status": "no negative-ratio observations" if term in omitted else "ok",
                                     "r_squared": 1 - np.square(residual).sum()/total if total > 0 else np.nan})
    coefficients = pd.DataFrame(coefficients)
    summary = coefficients.groupby(["model", "term"], sort=False, as_index=False).agg(
        mean=("coefficient", "mean"), std=("coefficient", "std"), n_years=("coefficient", "count"),
        min_stocks=("n_stocks", "min"), max_stocks=("n_stocks", "max"))
    summary["se"] = summary["std"] / np.sqrt(summary["n_years"])
    summary["t"] = (summary["mean"] / summary["se"]).where(summary["se"].gt(0))
    summary["n_total_years"] = formation["formation_year"].nunique()
    summary["n_unidentified_years"] = summary["n_total_years"] - summary["n_years"]
    if summary["n_years"].lt(2).any():
        raise ValueError("Table IV needs at least two identified annual coefficients per term for inference.")
    return summary.drop(columns="std"), coefficients
