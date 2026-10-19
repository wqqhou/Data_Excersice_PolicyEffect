"""Checks that cleaning preserves the analysis sample and state-month matches."""

from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from scripts.data_clean import merge_data, validate_data


def make_inputs():
    individuals = pd.DataFrame(
        {
            "person_id": [1, 2, 2, 1, 3],
            "state_name": ["Utah", "Kansas", "Kansas", "Utah", "Colorado"],
            "date": pd.to_datetime([
                "2016-03-01", "2016-02-01", "2016-03-01", "2016-02-01", "2016-03-01"
            ]),
            "age": [30, 41, 41, 30, 55],
            "college": [1, 0, 0, 1, 1],
            "hours_worked": [40, 0, 35, 40, 30],
            "wages": [2500, 0, 2200, 2400, 3000],
        },
        index=[10, 4, 99, 20, 7],
    )
    individuals["year"] = individuals["date"].dt.year
    individuals["month"] = individuals["date"].dt.month
    states = individuals[["state_name", "date", "year", "month"]].copy()
    states["population_size"] = [100000, 200000, 210000, 110000, 300000]
    states["employment_rate"] = [64, 65, 66, 67, 68]
    states["average_wage"] = [3100, 3200, 3300, 3400, 3500]
    states["minimum_wage"] = [7.25, 7.25, 7.25, 7.25, 8.31]
    return individuals, states.sample(frac=1, random_state=19)


class DataCleanTests(unittest.TestCase):
    def test_merge_keeps_observation_sample_order_and_state_covariates(self):
        individuals, states = make_inputs()
        original_individuals = individuals.copy(deep=True)
        original_states = states.copy(deep=True)

        merged = merge_data(individuals, states)
        expected = individuals.loc[individuals["wages"] > 0].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            merged[individuals.columns].reset_index(drop=True), expected
        )
        # A zero-earnings month must not remove this person's other months.
        self.assertEqual(merged.loc[merged["person_id"].eq(2), "wages"].tolist(), [2200])
        state_lookup = states.set_index(["state_name", "date"])
        for row in merged.itertuples(index=False):
            state_row = state_lookup.loc[(row.state_name, row.date)]
            for column in ("population_size", "employment_rate", "average_wage", "minimum_wage"):
                self.assertEqual(getattr(row, column), state_row[column])
        np.testing.assert_allclose(merged["ln_wages"], np.log(expected["wages"]))
        self.assertEqual(merged["treatment"].tolist(), [1, 1, 1, 0])
        self.assertEqual(merged["post"].tolist(), [1, 1, 0, 1])
        pd.testing.assert_frame_equal(individuals, original_individuals)
        pd.testing.assert_frame_equal(states, original_states)

    def test_merge_rejects_duplicate_state_months(self):
        individuals, states = make_inputs()
        states = pd.concat([states, states.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            merge_data(individuals, states)

    def test_merge_rejects_unmatched_individual_rows(self):
        individuals, states = make_inputs()
        states = states.loc[states["state_name"].ne("Colorado")]
        with self.assertRaises(ValueError):
            merge_data(individuals, states)

    def test_validation_removes_exact_state_duplicate_without_mutating_inputs(self):
        individuals, states = make_inputs()
        states = pd.concat([states, states.iloc[[0]]], ignore_index=True)
        original_individuals = individuals.copy(deep=True)
        original_states = states.copy(deep=True)

        with redirect_stdout(StringIO()):
            clean_individuals, clean_states = validate_data(individuals, states)

        self.assertEqual(len(clean_states), len(states) - 1)
        self.assertFalse(clean_states.duplicated(["state_name", "year", "month"]).any())
        expected_states = states.iloc[:-1].copy()
        expected_states["employment_rate"] /= 100
        pd.testing.assert_frame_equal(clean_states, expected_states)
        pd.testing.assert_frame_equal(clean_individuals, individuals)
        pd.testing.assert_frame_equal(individuals, original_individuals)
        pd.testing.assert_frame_equal(states, original_states)


if __name__ == "__main__":
    unittest.main()
