# Replication

Use Python 3.11 and place both supplied CSV files in `data/raw/`. From the project directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python replication.py
```

Rebuilds cleaned data, four plots, balance and regression tables, mover counts in both directions, inference checks, and `main.pdf`. PDF compilation requires LaTeX (`latexmk` or `pdflatex`); use `.venv/bin/python replication.py --skip-pdf` to omit it. No `.env` file is needed.
