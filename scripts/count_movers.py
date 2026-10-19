"""Q1(g): count moves between treatment and comparison states."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_STATES = {"Colorado", "Montana", "Wyoming"}
TREATED_STATES = {"Kansas", "Utah"}


def movers_between(data, origin_states, destination_states):
    # Compare consecutive observed residences within each person.
    history = data[["person_id", "date", "state_name"]].sort_values(
        ["person_id", "date"]
    ).copy()
    history["previous_state"] = history.groupby("person_id")["state_name"].shift()
    history["previous_date"] = history.groupby("person_id")["date"].shift()
    moved = (
        history["previous_state"].isin(origin_states)
        & history["state_name"].isin(destination_states)
    )
    # Count each person once per direction.
    return history.loc[moved].drop_duplicates("person_id").reset_index(drop=True)


def main():
    # Include zero-earnings months in the mover check.
    data = pd.read_parquet(PROJECT_ROOT / "data/interim/validated_idv.parquet")
    into_treatment = movers_between(data, CONTROL_STATES, TREATED_STATES)
    out_of_treatment = movers_between(data, TREATED_STATES, CONTROL_STATES)
    start, end = data["date"].min(), data["date"].max()
    sentence = (
        f"Between {start:%B %Y} and {end:%B %Y}, {len(into_treatment)} individuals moved "
        "at least once from a control state (Colorado, Montana, or Wyoming) "
        f"to a treated state (Kansas or Utah), and {len(out_of_treatment)} moved "
        "in the opposite direction, counting each person once per direction "
        "in the full validated sample, including zero-earnings observations."
    )
    output_dir = PROJECT_ROOT / "output/movers"
    output_dir.mkdir(parents=True, exist_ok=True)
    into_treatment.to_csv(output_dir / "control_to_treated.csv", index=False)
    out_of_treatment.to_csv(output_dir / "treated_to_control.csv", index=False)
    (output_dir / "mover_summary.tex").write_text(sentence + "\n", encoding="utf-8")
    print(sentence)


if __name__ == "__main__":
    main()
