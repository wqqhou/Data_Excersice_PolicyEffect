import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the individual wage distribution.")

    parser.add_argument("--interim-dir", default="data/interim",
                        help="Directory of interim files (default: data/interim)")
    parser.add_argument("--output-dir", default="output/plot",
                        help="Directory for output plot files (default: output/plot)")
    args = parser.parse_args()

    return args

def plot_1(df, output_path):
    """Save a histogram of individual wages with a numeric dollar axis."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x="wages", bins=50, ax=ax)
    ax.set_xlabel("Wage", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.set_title("Distribution of Wages (Current Dollar, Unemployed Included)", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path / "figure1_wage_distribution.png", dpi=300)
    plt.close(fig)

def plot_2(df, output_path):
    """Save a histogram of individual wages with a numeric dollar axis."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x="ln_wages", bins=50, ax=ax)
    ax.set_xlabel("Log Wage", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.set_title("Distribution of Log Wages (Current Dollar, Unemployed Excluded)", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path / "figure2_log_wage_distribution.png", dpi=300)
    plt.close(fig)

def main():
    args = parse_args()
    load_dotenv()
    base_path = Path(os.getenv("BASE_PROJECT_PATH"))
    p1_dir = base_path / args.interim_dir / "validated_idv.parquet"
    p2_dir = base_path / args.interim_dir / "p2.parquet"
    output_dir = base_path / args.output_dir
    df_p1 = pd.read_parquet(p1_dir)
    df_p2 = pd.read_parquet(p2_dir)
    plot_1(df_p1, output_dir)
    plot_2(df_p2, output_dir)


if __name__ == "__main__":
    main()
