"""Q1(a), (b), and (d): descriptive figures."""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interim-dir", type=Path, default=Path("data/interim"),
        help="Interim files, relative to the project root (default: data/interim).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/plot"),
        help="Plot directory, relative to the project root (default: output/plot).",
    )
    return parser.parse_args()


def plot_1(df, output_path):
    """Q1(a.i): monthly earnings distribution."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x="wages", bins=50, ax=ax)
    ax.set_xlabel("Wage", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.set_title(
        "Distribution of Wages (Current Dollar, Unemployed Included)",
        fontsize=16, fontweight="bold", pad=15,
    )
    fig.tight_layout()
    fig.savefig(output_path / "figure1_wage_distribution.png", dpi=300)
    plt.close(fig)


def plot_2(df, output_path):
    """Q1(a.ii): log monthly earnings distribution."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x="ln_wages", bins=50, ax=ax)
    ax.set_xlabel("Log Wage", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.set_title(
        "Distribution of Log Wages (Current Dollar, Unemployed Excluded)",
        fontsize=16, fontweight="bold", pad=15,
    )
    fig.tight_layout()
    fig.savefig(output_path / "figure2_log_wage_distribution.png", dpi=300)
    plt.close(fig)


def plot_3(df, output_path):
    """Q1(b): mean earnings by weekly hours and state."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.lineplot(
        data=df,
        x="hours_worked",
        y="wages",
        hue="state_name",
        hue_order=sorted(df["state_name"].unique()),
        estimator="mean",
        errorbar=None,
        marker="o",
        sort=True,
        ax=ax,
    )
    ax.set_xlabel("Weekly work hours", fontsize=12, labelpad=10)
    ax.set_ylabel("Average monthly wage", fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.set_title(
        "Average monthly wage by weekly hours worked and state",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.legend(title="State", ncol=5, loc="upper left")
    fig.text(
        0.5, 0.015,
        "Each point averages individual monthly earnings at the indicated hours worked.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path / "figure3_wage_dif_workhour.png", dpi=300)
    plt.close(fig)


def plot_4(df, output_path, baseline="2016-02-01"):
    """Q1(d): log earnings and hours for the baseline-employed cohort."""
    policy_date = pd.Timestamp("2016-03-01")
    baseline_date = pd.Timestamp(baseline).to_period("M").to_timestamp()
    if pd.isna(baseline_date) or baseline_date >= policy_date:
        raise ValueError("The baseline must be a valid month before March 2016.")

    columns = ["person_id", "date", "state_name", "wages", "ln_wages", "hours_worked"]
    data = df[columns].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    if data[["person_id", "date"]].isna().any().any():
        raise ValueError("Dates and person_id cannot be missing.")
    if data.duplicated(["person_id", "date"]).any():
        raise ValueError("plot_4 requires one observation per person and month.")

    treated_states = {"Kansas", "Utah"}
    comparison_states = {"Colorado", "Montana", "Wyoming"}
    groups = ["Kansas / Utah", "Colorado / Montana / Wyoming"]
    employed = data.loc[data["wages"].gt(0) & data["hours_worked"].gt(0)].copy()
    base = employed.loc[employed["date"].eq(baseline_date), ["person_id", "state_name"]].copy()
    if base.empty:
        raise ValueError(f"No employed baseline observations for {baseline_date:%Y-%m}.")
    if not base["state_name"].isin(treated_states | comparison_states).all():
        raise ValueError("Baseline residence must be one of the five supplied states.")
    base["group"] = base["state_name"].isin(treated_states).map(
        {True: groups[0], False: groups[1]}
    )
    if base["group"].nunique() != 2:
        raise ValueError("The baseline sample must contain both state groups.")

    # Keep movers in their baseline residence group.
    sample = employed.merge(
        base[["person_id", "group"]], on="person_id", how="inner", validate="many_to_one"
    )
    # Average observed employed members within each month and group.
    monthly = sample.groupby(["group", "date"], as_index=False).agg(
        mean_log_earnings=("ln_wages", "mean"),
        mean_hours=("hours_worked", "mean"),
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    panels = [
        ("mean_log_earnings", "Panel A: Mean log monthly earnings", "Log monthly earnings (natural log)"),
        ("mean_hours", "Panel B: Mean weekly hours", "Weekly hours"),
    ]
    colors = ["#dd8452", "#4c72b0"]
    for ax, (variable, title, ylabel) in zip(axes, panels):
        for group, color in zip(groups, colors):
            series = monthly.loc[monthly["group"].eq(group)].sort_values("date")
            ax.plot(series["date"], series[variable], label=group, color=color, linewidth=2)
        ax.axvline(
            policy_date, color="0.3", linestyle="--", linewidth=1.5, label="March 2016"
        )
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle("Monthly outcomes by baseline state group", fontsize=16, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=3, frameon=False,
    )
    fig.text(
        0.5, 0.02,
        f"Groups fixed by {baseline_date:%B %Y} residence; movers retain their baseline group.\n"
        "Includes people employed at baseline; each month's means use those observed employed.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(output_path / "figure4_monthly_earnings_hours.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    interim_dir = PROJECT_ROOT / args.interim_dir
    output_dir = PROJECT_ROOT / args.output_dir
    df_individual = pd.read_parquet(interim_dir / "validated_idv.parquet")
    df_merged = pd.read_parquet(interim_dir / "merged.parquet")
    plot_1(df_individual, output_dir)
    plot_2(df_merged, output_dir)
    plot_3(df_merged, output_dir)
    plot_4(df_merged, output_dir)


if __name__ == "__main__":
    main()
