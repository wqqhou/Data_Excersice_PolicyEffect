"""Compare baseline characteristics of treated and comparison-state residents."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DATE = pd.Timestamp("2016-03-01")
TREATED_STATES = {"Kansas", "Utah"}
CONTROL_STATES = {"Colorado", "Montana", "Wyoming"}
VARIABLES = {
    "age": ("Age (years)", 1.0),
    "college": ("College (%)", 100.0),
    "wages": ("Monthly earnings (USD)", 1.0),
    "hours_worked": ("Weekly hours", 1.0),
}


def _clustered_difference(sample):
    treated = sample.loc[sample["treatment"].eq(1), "value"]
    control = sample.loc[sample["treatment"].eq(0), "value"]
    difference = treated.mean() - control.mean()
    state_counts = sample.groupby("treatment")["state_name"].nunique()
    n, g = len(sample), sample["state_name"].nunique()

    if min(len(treated), len(control)) < 2 or state_counts.min() < 2:
        return difference, np.nan, np.nan

    variance = 0.0
    for arm in (0, 1):
        group = sample.loc[sample["treatment"].eq(arm)]
        residuals = group["value"] - group["value"].mean()
        state_residual_sums = residuals.groupby(group["state_name"]).sum()
        variance += np.square(state_residual_sums).sum() / len(group) ** 2

    # Apply the finite-sample correction (K=2).
    variance *= (g / (g - 1)) * ((n - 1) / (n - 2))
    se = np.sqrt(variance)
    scale = max(1.0, sample["value"].abs().max())
    if not np.isfinite(se) or se <= np.finfo(float).eps * scale:
        return difference, np.nan, np.nan
    p_value = 2 * stats.t.sf(abs(difference / se), df=g - 1)
    return difference, se, p_value


def balance_test(df, date_col="date", baseline="2016-02-01", employed_only=True):
    data = df.copy()
    required = {"person_id", "state_name", *VARIABLES}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize dates before validation.
    if date_col in data.columns:
        dates = pd.to_datetime(data[date_col], errors="raise")
    elif {"year", "month"}.issubset(data.columns):
        dates = pd.to_datetime(data[["year", "month"]].assign(day=1))
    else:
        raise ValueError(f"Need '{date_col}' or both 'year' and 'month'.")
    if dates.isna().any() or data["person_id"].isna().any():
        raise ValueError("Dates and person_id cannot be missing.")
    data["date"] = dates.dt.to_period("M").dt.to_timestamp()
    if data.duplicated(["person_id", "date"]).any():
        raise ValueError(
            "Duplicate person-month records found. Check your merge; the supplied "
            "state file has an exact duplicate for Kansas in March 2016. "
            "Resolve the source duplicate and validate a many-to-one merge."
        )

    data["state_name"] = data["state_name"].astype("string").str.strip()
    allowed = TREATED_STATES | CONTROL_STATES
    if not data["state_name"].isin(allowed).all():
        raise ValueError("Missing or unexpected state names; expected the five supplied states.")
    for variable in VARIABLES:
        data[variable] = pd.to_numeric(data[variable], errors="raise")
    if not data["college"].dropna().isin([0, 1]).all():
        raise ValueError("college must be coded 0/1 (or missing), not percentages.")
    if data[["wages", "hours_worked"]].lt(0).any().any():
        raise ValueError("Wages/hours cannot be negative in these supplied data.")

    # Assign groups using baseline residence.
    data["treatment"] = data["state_name"].isin(TREATED_STATES).astype(int)
    data["post"] = data["date"].ge(POLICY_DATE).astype(int)
    data["treated_post"] = data["treatment"] * data["post"]
    baseline_date = pd.Timestamp(baseline).to_period("M").to_timestamp()
    if pd.isna(baseline_date) or baseline_date >= POLICY_DATE:
        raise ValueError("The baseline must be a valid month before March 2016.")
    base = data.loc[data["date"].eq(baseline_date)].copy()
    if base.empty:
        raise ValueError(f"No observations for {baseline_date:%Y-%m}.")
    n_before = len(base)

    # Keep employed baseline observations.
    if employed_only:
        base = base.loc[base["wages"].gt(0) & base["hours_worked"].gt(0)].copy()
    if base["treatment"].nunique() != 2:
        raise ValueError("The baseline sample must contain both treatment and control groups.")

    rows = []
    # Use each variable's nonmissing sample.
    for variable, (label, multiplier) in VARIABLES.items():
        sample = base[["state_name", "treatment", variable]].rename(
            columns={variable: "value"}
        ).copy()
        sample["value"] = sample["value"].replace([np.inf, -np.inf], np.nan)
        sample = sample.dropna(subset=["value"])
        sample["value"] *= multiplier
        t = sample.loc[sample["treatment"].eq(1), "value"]
        c = sample.loc[sample["treatment"].eq(0), "value"]
        if t.empty or c.empty:
            raise ValueError(f"{variable}: no nonmissing observations in one group.")
        difference, se, p_value = _clustered_difference(sample)
        pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        rows.append({
            "variable": label,
            "n_treated": len(t), "n_control": len(c),
            "mean_treated": t.mean(), "mean_control": c.mean(),
            "sd_treated": t.std(ddof=1), "sd_control": c.std(ddof=1),
            "difference_t_minus_c": difference,
            "standardized_difference": difference / pooled_sd if pooled_sd > 0 else np.nan,
            "se_state_cluster_cr1": se,
            "p_state_cluster_approx": p_value,
            "states_treated": sample.loc[sample["treatment"].eq(1), "state_name"].nunique(),
            "states_control": sample.loc[sample["treatment"].eq(0), "state_name"].nunique(),
            "df_t_reference": sample["state_name"].nunique() - 1,
        })

    sample_description = (
        "Employed at baseline (wages > 0 and hours > 0)"
        if employed_only
        else "All supplied baseline observations, including nonemployment"
    )
    notes = (
        f"Baseline: {baseline_date:%B %Y}. One observation per person. {sample_description}.\n"
        f"Input baseline N={n_before}; analysis baseline N={len(base)}; "
        f"excluded by employment filter={n_before - len(base)}.\n"
        "Treatment group: Kansas/Utah; comparison: Colorado/Montana/Wyoming, using baseline residence.\n"
        "Differences are treated minus control, using person-weighted means. "
        "College is in percent; its difference is in percentage points.\n"
        "H0 for each row: equal pre-policy group means; H1: unequal means (two-sided).\n"
        "SMD = difference / sqrt((SD_treated^2 + SD_control^2)/2). "
        "N and state counts reflect each variable's nonmissing sample.\n"
        "SEs: state-clustered CR1 for OLS on an intercept and treatment indicator, "
        "with G/(G-1)*(N-1)/(N-2) correction. P-values use t(G-1).\n"
        "CAUTION: p-values are approximate. Five states, including only two treated states, "
        "are too few for dependable conventional clustered inference.\n"
        "Unavailable inference is reported as NA if there are fewer than two states per arm, "
        "too few observations, or a zero/negligible SE.\n"
        "Failure to reject does not establish equivalence or parallel trends. "
        "Baseline mean differences do not by themselves invalidate DiD.\n"
    )
    return pd.DataFrame(rows), notes


def save_results(table, notes, output_dir="balance_results"):
    # Export CSV, text, and LaTeX tables.
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "balance_table.csv", index=False, na_rep="NA", float_format="%.8g")
    display_cols = [
        "variable", "n_treated", "n_control", "mean_treated", "mean_control",
        "difference_t_minus_c", "standardized_difference", "p_state_cluster_approx",
    ]
    report = table[display_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}", na_rep="NA"
    )
    (destination / "balance_report.txt").write_text(notes + "\n" + report + "\n", encoding="utf-8")

    latex = [
        r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Pre-policy balance}", r"\label{tab:baseline_balance}",
        r"\begin{tabular}{lrrrrr}", r"\hline",
        r"Variable & Treated mean (SD) & Control mean (SD) & Difference & SMD & Approx. $p$ \\",
        r"\hline",
    ]
    def fmt(value):
        return "NA" if pd.isna(value) else f"{value:,.2f}"

    for row in table.itertuples(index=False):
        p = row.p_state_cluster_approx
        p_text = "NA" if pd.isna(p) else (r"$<0.001$" if p < 0.001 else f"{p:.3f}")
        label = row.variable.replace("%", r"\%")
        latex.append(
            f"{label} & {fmt(row.mean_treated)} ({fmt(row.sd_treated)}) & "
            f"{fmt(row.mean_control)} ({fmt(row.sd_control)}) & "
            f"{fmt(row.difference_t_minus_c)} & {fmt(row.standardized_difference)} & {p_text} "
            + r"\\"
        )
    latex.extend([
        r"\hline", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}", r"\footnotesize",
    ])
    latex.append(notes.splitlines()[0].replace("%", r"\%").replace(">", r"$>$"))
    latex.append(
        " Treatment group: Kansas/Utah; control group: Colorado/Montana/Wyoming. "
        "Differences are treated minus control. College differences are percentage points."
    )
    latex.append(
        " State-clustered CR1 tests use a conventional small-sample correction "
        "and a $t_{G-1}$ reference. P-values are approximate; inference is fragile "
        "with five states and only two treated states."
    )
    latex.append(
        " SMD uses the square root of the average of the two within-group variances. "
        "Failure to reject does not establish equivalence or parallel trends."
    )
    count_entries = []
    for row in table.itertuples(index=False):
        label = row.variable.replace("%", r"\%")
        count_entries.append(f"{label}: {row.n_treated}/{row.n_control}")
    counts = "; ".join(count_entries)
    latex.append(" Sample sizes (treated/control): " + counts + ".")
    latex.extend([r"\end{minipage}", r"\end{table}"])
    (destination / "balance_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("merged.parquet"),
        help="Input file, relative to data/interim, or an absolute path.",
    )
    parser.add_argument("--date-col", default="date", help="Existing datetime column, if present.")
    parser.add_argument(
        "--baseline", default="2016-02-01", help="Any baseline month strictly before March 2016."
    )
    parser.add_argument("--include-nonemployed", action="store_true", help="Keep baseline zero wages/hours.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("balance_results"),
        help="Output directory, relative to the project root, or an absolute path.",
    )
    args = parser.parse_args()

    merged_path = PROJECT_ROOT / "data/interim" / args.input
    output_dir = PROJECT_ROOT / args.output_dir
    try:
        table, notes = balance_test(
            pd.read_parquet(merged_path), date_col=args.date_col,
            baseline=args.baseline, employed_only=not args.include_nonemployed,
        )
        report = save_results(table, notes, output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(notes)
    print(report)
    print(
        "\nSaved balance_table.csv, balance_report.txt, and balance_table.tex "
        f"to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
