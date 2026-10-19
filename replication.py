"""Run the analysis and compile the report."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = (
    ("Clean and merge data", "data_clean.py"),
    ("Generate four figures", "plot.py"),
    ("Generate baseline balance table", "balance_test.py"),
    ("Estimate and verify DiD regressions", "estimate.py"),
    ("Count movers in both treatment directions", "count_movers.py"),
)


def run_replication(*, skip_pdf=False):
    required = [
        PROJECT_ROOT / "data/raw/monthly_individual_data.csv",
        PROJECT_ROOT / "data/raw/monthly_state_data.csv",
    ]
    pdf_commands = []
    if not skip_pdf:
        required.append(PROJECT_ROOT / "main.tex")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    if not skip_pdf:
        latexmk = shutil.which("latexmk")
        pdflatex = shutil.which("pdflatex")
        options = ["-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        if latexmk:
            pdf_commands = [[latexmk, "-pdf", "-quiet", *options]]
        elif pdflatex:
            # Compile twice to resolve cross-references.
            pdf_commands = [[pdflatex, *options], [pdflatex, *options]]
        else:
            raise RuntimeError(
                "PDF compilation requires latexmk or pdflatex on PATH. "
                "Install a LaTeX distribution, or use --skip-pdf for data, figures, and tables."
            )

    with tempfile.TemporaryDirectory(prefix="metcalfe-mpl-") as cache_dir:
        env = dict(
            os.environ,
            MPLBACKEND="Agg",
            MPLCONFIGDIR=cache_dir,
            PYTHONUNBUFFERED="1",
        )
        for label, script in STAGES:
            print(f"\n=== {label} ===", flush=True)
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
        if pdf_commands:
            print("\n=== Compile main.pdf ===", flush=True)
            for command in pdf_commands:
                subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    outputs = "data/interim/, output/plot/, balance_results/, output/regression/, output/movers/"
    print(f"\nReplication complete. Outputs: {outputs}", flush=True)
    if not skip_pdf:
        print(f"PDF: {PROJECT_ROOT / 'main.pdf'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Rebuild all data, figures, and tables without running LaTeX.",
    )
    args = parser.parse_args()
    try:
        run_replication(skip_pdf=args.skip_pdf)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode or 1, f"\nReplication stopped: stage failed ({exc.cmd[0]}).\n")
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"\nReplication stopped: {exc}\n")


if __name__ == "__main__":
    main()
