import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

US_STATES = set("Colorado Montana Kansas Utah Wyoming".split())

def parse_args():
    parser = argparse.ArgumentParser(description="Process monthly_individual_data.")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="First year to process (default: 2015)")
    parser.add_argument("--start-month", type=int, choices=range(1, 13), default=1,
                        help="First month to process (default: 1)")

    parser.add_argument("--end-year", type=int, default=2017,
                        help="Last year to process (default: 2017)")
    parser.add_argument("--end-month", type=int, choices=range(1, 13), default=12,
                        help="Last month to process (default: 12)")

    parser.add_argument("--interim-dir", default="data/interim",
                        help="Directory of extracted files (default: data/interim)")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory for raw files (default: data/raw)")

    args = parser.parse_args()
    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        parser.error("Start year/month must be on or before end year/month")
    return args

def read_data(extract_dir, start_year, end_year, start_month=1, end_month=12):
    """Read both datasets, keeping all months in the inclusive date range."""
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1)
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date")

    idv_file = Path(extract_dir) / "monthly_individual_data.csv"
    state_file = Path(extract_dir) / "monthly_state_data.csv"
    if not idv_file.exists():
        raise FileNotFoundError(f"Missing individual data: {idv_file}")
    if not state_file.exists():
        raise FileNotFoundError(f"Missing state data: {state_file}")
    idv_columns = ["person_id","year","month","state_name","age","college","hours_worked","wages"]
    state_columns = ["state_name","year","month","population_size","employment_rate","average_wage","minimum_wage"]
    df_idv = pd.read_csv(idv_file, encoding="cp1252", low_memory=False, usecols=idv_columns, dtype={"state_name": "string"})
    df_state = pd.read_csv(state_file, encoding="cp1252", low_memory=False, usecols=state_columns, dtype={"state_name": "string"})

    for df in (df_idv, df_state):
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    df_idv = df_idv.loc[df_idv["date"].between(start_date, end_date, inclusive="both")].copy()
    df_state = df_state.loc[df_state["date"].between(start_date, end_date, inclusive="both")].copy()
    return df_idv, df_state

def _report_missing(df, label):
    """Report missing cells by column and the number of affected rows."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    affected_rows = int(df.isna().any(axis=1).sum())
    print(f"{label}: {affected_rows} row(s) with missing values")
    for column, count in missing.items():
        print(f"  {column}: {int(count)} missing value(s)")

def _apply_validation_rules(df, label, rules, duplicate_keys):
    """Count each removed row once; keep the first remaining row per key."""
    keep = pd.Series(True, index=df.index)
    removals = []
    for description, valid in rules.items():
        valid = valid.fillna(False)
        count = int((keep & ~valid).sum())
        removals.append((description, count))
        keep &= valid

    cleaned = df.loc[keep].copy()
    duplicates = cleaned.duplicated(subset=duplicate_keys, keep="first")
    removals.append((f"duplicate ({', '.join(duplicate_keys)})", int(duplicates.sum())))
    cleaned = cleaned.loc[~duplicates].copy()

    print(f"{label}: removed {len(df) - len(cleaned)} of {len(df)} row(s); "
          f"retained {len(cleaned)}")
    for description, count in removals:
        if count:
            print(f"  {description}: {count} removed")
    _report_missing(cleaned, f"{label} after validation")
    return cleaned

def validate_data(df_idv, df_state):
    df_idv = df_idv.copy()
    df_state = df_state.copy()
    idv_numeric = ["person_id", "age", "college", "hours_worked", "wages"]
    state_numeric = ["employment_rate", "population_size", "average_wage", "minimum_wage"]

    for label, frame, numeric_columns in (
        ("df_idv", df_idv, idv_numeric),
        ("df_state", df_state, state_numeric),
    ):
        _report_missing(frame, f"{label} before validation")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # The source CSV stores employment rates as percentages (e.g. 64).
    df_state["employment_rate"] = df_state["employment_rate"] / 100

    idv_rules = {
        "person_id must be numeric and finite": df_idv["person_id"].notna() & np.isfinite(df_idv["person_id"]),
        "state_name must be one of the five allowed states": df_idv["state_name"].isin(US_STATES),
        "age must be in (0, 150)": df_idv["age"].between(0, 150, inclusive="neither"),
        "college must be 0 or 1": df_idv["college"].isin([0, 1]),
        "hours_worked must be in [0, 168)": df_idv["hours_worked"].between(0, 168, inclusive="left"),
        "wages must be in [0, +inf)": df_idv["wages"].ge(0) & np.isfinite(df_idv["wages"]),
    }
    state_rules = {
        "employment_rate must be in (0, 1)": df_state["employment_rate"].between(0, 1, inclusive="neither"),
        "population_size must be positive and finite": df_state["population_size"].gt(0) & np.isfinite(df_state["population_size"]),
        "average_wage must be positive and finite": df_state["average_wage"].gt(0) & np.isfinite(df_state["average_wage"]),
        "minimum_wage must be positive and finite": df_state["minimum_wage"].gt(0) & np.isfinite(df_state["minimum_wage"]),
    }
    df_idv = _apply_validation_rules(df_idv, "df_idv", idv_rules, ["person_id", "date"])
    df_state = _apply_validation_rules(df_state, "df_state", state_rules, ["state_name", "year", "month"])
    return df_idv, df_state

def merge_data(df_idv, df_state):
    merged_df = df_state.join(df_idv, how="left").reset_index()
    return merged_df

def main():
    args = parse_args()
    load_dotenv()
    base_path = Path(os.getenv("BASE_PROJECT_PATH"))

    raw_dir = base_path / args.raw_dir
    interim_dir = base_path / args.interim_dir
    idv_path = interim_dir / "validated_idv.parquet"
    state_path = interim_dir / "validated_state.parquet"
    p2_path = interim_dir / "p2.parquet"

    df_idv, df_state = read_data(
        raw_dir, args.start_year, args.end_year, args.start_month, args.end_month
    )
    df_idv, df_state = validate_data(df_idv, df_state)
    df_idv.to_parquet(idv_path, index=False)
    df_state.to_parquet(state_path, index=False)

    df_p2 = df_idv.loc[df_idv["wages"]>0]
    df_p2["ln_wages"] = np.log(df_p2["wages"])
    df_p2.to_parquet(p2_path, index=False)

    print(f"Validated DF saved as parquet at: {interim_dir}")




if __name__ == "__main__":
    main()