"""Q1(e): state and calendar-month fixed-effects DiD with state-clustered inference."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {
    "ln_wages": "Log monthly earnings",
    "hours_worked": "Weekly hours",
}
POLICY_TERM = "treated_post"
INFERENCE_NOTE = (
    "SEs are clustered by state. The conventional finite-sample covariance correction "
    "G/(G-1) * (N-1)/(N-K) is applied once. Two-sided p-values and 95% confidence "
    "intervals use Student's t with G-1 degrees of freedom. Inference remains "
    "approximate and potentially unreliable with only five states and two treated states."
)
MODEL_NOTE = (
    "Specification: Y_ist = state FE + calendar-month FE + beta*(treatment*post) + error. "
    "Outcomes are the existing ln_wages and hours_worked (levels). Input is the "
    "positive-wage sample in merged.parquet; only missing/nonfinite model fields are "
    "excluded separately for each outcome. Stored treatment and post indicators are "
    "used unchanged: Kansas/Utah by current residence, post from March 2016. Movers "
    "follow the stored current-state assignment; no baseline-cohort restriction is added. "
    "K counts the intercept, policy coefficient, G-1 state dummies, and T-1 full "
    "calendar-month dummies. Regression residual df remains N-K."
)


@dataclass
class DidFit:
    outcome: str
    sample: pd.DataFrame
    ordinary: object
    result: object
    correction_factor: float


def fit_did(df, outcome):
    # Retain complete model rows and their original positions.
    required = [outcome, "state_name", "date", "treatment", "post"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    sample = df.copy().reset_index(drop=True)
    sample["source_row"] = np.arange(len(sample))
    sample["date"] = pd.to_datetime(sample["date"], errors="raise")
    for column in (outcome, "treatment", "post"):
        sample[column] = pd.to_numeric(sample[column], errors="raise")
        sample[column] = sample[column].replace([np.inf, -np.inf], np.nan)
    sample = sample.dropna(subset=required).copy()
    if sample.empty:
        raise ValueError(f"No complete observations for {outcome}.")
    for column in ("treatment", "post"):
        if not sample[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0/1.")

    sample["calendar_month"] = sample["date"].dt.to_period("M").astype(str)
    sample[POLICY_TERM] = sample["treatment"] * sample["post"]
    # Build state and calendar-month fixed effects.
    fixed_effects = pd.get_dummies(
        sample[["state_name", "calendar_month"]].astype(str),
        drop_first=True, dtype=float,
    )
    design = sm.add_constant(
        pd.concat([sample[[POLICY_TERM]].astype(float), fixed_effects], axis=1),
        has_constant="add",
    )
    n, k = design.shape
    g = sample["state_name"].nunique()
    if g < 2:
        raise ValueError("State-clustered inference requires at least two states.")
    if n <= k or np.linalg.matrix_rank(design.to_numpy()) != k:
        raise ValueError("The regression design must have full column rank and N > K.")

    model = sm.OLS(sample[outcome].astype(float), design, missing="raise")
    ordinary = model.fit()
    # Fit with state clusters and t(G-1) inference.
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": sample["state_name"],
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )
    # Recorded only; model.fit already applies this correction.
    correction_factor = g / (g - 1) * (n - 1) / (n - k)
    return DidFit(outcome, sample, ordinary, result, correction_factor)


def policy_row(fit):
    result = fit.result
    lower, upper = result.conf_int(alpha=0.05).loc[POLICY_TERM]
    return {
        "outcome": fit.outcome,
        "outcome_label": OUTCOMES.get(fit.outcome, fit.outcome),
        "coefficient": float(result.params[POLICY_TERM]),
        "std_error": float(result.bse[POLICY_TERM]),
        "t_statistic": float(result.tvalues[POLICY_TERM]),
        "p_value": float(result.pvalues[POLICY_TERM]),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "nobs": int(result.nobs),
        "n_clusters": int(fit.sample["state_name"].nunique()),
        "inference_df": int(result.df_resid_inference),
        "n_coefficients": len(result.params),
        "residual_df": int(result.df_resid),
        "correction_factor": fit.correction_factor,
    }


def verify_fit(fit):
    result, ordinary, sample = fit.result, fit.ordinary, fit.sample
    x = result.model.exog
    y = result.model.endog
    n, k = x.shape
    states = sample["state_name"].to_numpy()
    clusters = np.unique(states)
    g = len(clusters)
    np.testing.assert_array_equal(result.model.data.row_labels, sample.index)
    np.testing.assert_array_equal(result.cov_kwds["groups"], states)
    np.testing.assert_array_equal(y, sample[fit.outcome].to_numpy())
    np.testing.assert_array_equal(ordinary.model.exog, x)
    np.testing.assert_array_equal(ordinary.model.endog, y)
    np.testing.assert_allclose(result.params, ordinary.params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.params, np.linalg.lstsq(x, y, rcond=None)[0], rtol=1e-10, atol=1e-10
    )
    if result.nobs != ordinary.nobs or result.df_resid != n - k:
        raise AssertionError("Observation count or regression residual df changed.")
    if not result.use_t or result.df_resid_inference != g - 1:
        raise AssertionError("Inference must use Student's t with G-1 df.")

    # Independently verify the covariance and t-based inference.
    residuals = y - x @ ordinary.params.to_numpy()
    scores = np.vstack([
        x[states == state].T @ residuals[states == state] for state in clusters
    ])
    bread = np.linalg.inv(x.T @ x)
    uncorrected_covariance = bread @ (scores.T @ scores) @ bread
    factor = g / (g - 1) * (n - 1) / (n - k)
    expected_covariance = uncorrected_covariance * factor
    np.testing.assert_allclose(result.cov_params(), expected_covariance, rtol=1e-8, atol=1e-10)
    position = result.params.index.get_loc(POLICY_TERM)
    coefficient = float(result.params[POLICY_TERM])
    se = float(np.sqrt(expected_covariance[position, position]))
    t_statistic = coefficient / se
    p_value = 2 * stats.t.sf(abs(t_statistic), df=g - 1)
    critical_value = stats.t.ppf(0.975, df=g - 1)
    interval = np.array([
        coefficient - critical_value * se, coefficient + critical_value * se
    ])
    row = policy_row(fit)
    np.testing.assert_allclose(
        [row["std_error"], row["t_statistic"], row["p_value"]],
        [se, t_statistic, p_value], rtol=1e-8, atol=1e-12,
    )
    np.testing.assert_allclose(
        [row["ci_lower"], row["ci_upper"]], interval, rtol=1e-8, atol=1e-10
    )
    return {
        "outcome": fit.outcome,
        "nobs": n,
        "n_clusters": g,
        "inference_df": g - 1,
        "n_coefficients": k,
        "correction_factor": factor,
        "cluster_labels_aligned": True,
        "coefficients_and_sample_match_plain_ols": True,
        "correction_applied_once": True,
        "scipy_t_pvalue_and_ci_match": True,
        "max_coefficient_change_vs_plain_ols": float(
            np.max(np.abs(result.params - ordinary.params))
        ),
        "max_covariance_difference": float(
            np.max(np.abs(result.cov_params().to_numpy() - expected_covariance))
        ),
        "independent_p_value": float(p_value),
        "independent_ci": interval.tolist(),
    }


def save_results(fits, output_dir):
    # Verify fits, then export tables, estimation rows, and checks.
    checks = [verify_fit(fit) for fit in fits]
    table = pd.DataFrame([policy_row(fit) for fit in fits])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "did_table.csv", index=False, float_format="%.17g")
    display = table.drop(columns=["outcome", "correction_factor", "residual_df"])
    report = display.to_string(index=False, float_format=lambda value: f"{value:.6g}")
    provenance = (
        "Implemented from the supplied Q1(e) equation; the previous estimate.py was empty. "
        "Verification compares clustered inference with identical ordinary OLS models, "
        "not with a previously saved DiD analysis."
    )
    (destination / "did_report.txt").write_text(
        f"Q1(e): difference-in-differences\n\n{report}\n\n{INFERENCE_NOTE}\n\n{MODEL_NOTE}\n\n{provenance}\n",
        encoding="utf-8",
    )
    latex = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Q1(e): Difference-in-differences estimates}",
        r"\label{tab:did}", r"\begin{tabular}{lrr}", r"\hline",
        r" & Log monthly earnings & Weekly hours \\", r"\hline",
    ]
    fields = [
        (r"Policy coefficient ($\hat\beta$)", "coefficient", ".6f"),
        ("State-clustered SE", "std_error", ".6f"),
        (r"$t$ statistic", "t_statistic", ".4f"),
        (r"Two-sided $p$-value", "p_value", ".6g"),
        (r"95\% CI lower", "ci_lower", ".6f"),
        (r"95\% CI upper", "ci_upper", ".6f"),
        (r"Observations ($N$)", "nobs", ",d"),
        (r"State clusters ($G$)", "n_clusters", "d"),
        (r"Inference df ($G-1$)", "inference_df", "d"),
        (r"Coefficients incl. FE ($K$)", "n_coefficients", "d"),
    ]
    for label, column, fmt in fields:
        values = " & ".join(format(value, fmt) for value in table[column])
        latex.append(f"{label} & {values} " + r"\\")
    latex.extend([
        r"State fixed effects & Yes & Yes \\",
        r"Calendar-month fixed effects & Yes & Yes \\",
        r"\hline", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}", r"\footnotesize",
        INFERENCE_NOTE.replace("95%", r"95\%"),
        "Positive-wage input sample; stored treatment and post indicators. "
        "K includes the intercept, policy coefficient, and estimated state and month dummies.",
        r"\end{minipage}", r"\end{table}",
    ])
    (destination / "did_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    for fit in fits:
        identity_columns = [
            column for column in (
                "source_row", "person_id", "date", "state_name", "treatment", "post", POLICY_TERM
            )
            if column in fit.sample
        ]
        fit.sample[identity_columns].to_csv(
            destination / f"{fit.outcome}_estimation_sample.csv", index=False
        )
    (destination / "verification.json").write_text(
        json.dumps({
            "statsmodels_version": statsmodels.__version__,
            "comparison": provenance,
            "models": checks,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/merged.parquet"),
        help="Input file, relative to the project root, or an absolute path.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/regression"),
        help="Output directory, relative to the project root, or an absolute path.",
    )
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    data = pd.read_parquet(PROJECT_ROOT / args.input)
    fits = [fit_did(data, outcome) for outcome in OUTCOMES]
    table = save_results(fits, output_dir)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\n{INFERENCE_NOTE}\n\nVerification passed. Results: {output_dir}")


if __name__ == "__main__":
    main()
