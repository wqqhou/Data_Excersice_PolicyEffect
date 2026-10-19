"""Checks for state-clustered inference without changing the fitted DiD model."""

import json
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from scripts.estimate import fit_did, policy_row, verify_fit


def make_panel():
    rng = np.random.default_rng(4101)
    states = ["Colorado", "Kansas", "Montana", "Utah", "Wyoming"]
    dates = pd.date_range("2015-01-01", periods=18, freq="MS")
    rows = []
    for state_number, state in enumerate(states):
        treated = int(state in {"Kansas", "Utah"})
        for month_number, date in enumerate(dates):
            post = int(date >= pd.Timestamp("2016-03-01"))
            state_month_shock = rng.normal(scale=0.1)
            for person in range(3):
                log_wage = (
                    8 + 0.03 * state_number + 0.02 * month_number
                    + 0.16 * treated * post + state_month_shock
                    + rng.normal(scale=0.08)
                )
                rows.append({
                    "person_id": state_number * 3 + person,
                    "state_name": state,
                    "date": date,
                    "month": date.month,
                    "treatment": treated,
                    "post": post,
                    "ln_wages": log_wage,
                    "wages": np.exp(log_wage),
                    "hours_worked": (
                        38 + state_number + 0.03 * month_number
                        + 0.4 * treated * post + 2 * state_month_shock
                        + rng.normal(scale=0.2)
                    ),
                })
    return pd.DataFrame(rows)


class EstimateTests(unittest.TestCase):
    def test_cluster_sandwich_and_exported_t_inference(self):
        data = make_panel()
        for outcome in ("ln_wages", "hours_worked"):
            with self.subTest(outcome=outcome):
                fit = fit_did(data, outcome)
                x = np.asarray(fit.ordinary.model.exog)
                y = np.asarray(fit.ordinary.model.endog)
                coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
                residuals = y - x @ coefficients
                bread = np.linalg.inv(x.T @ x)
                groups = fit.sample["state_name"].to_numpy()
                states = np.unique(groups)
                scores = np.stack([
                    x[groups == state].T @ residuals[groups == state]
                    for state in states
                ])
                n, k = x.shape
                g = len(states)
                correction = g / (g - 1) * (n - 1) / (n - k)
                expected_covariance = bread @ (scores.T @ scores) @ bread * correction

                # Repeated month numbers across years require distinct time FE.
                self.assertEqual(k, data["state_name"].nunique() + data["date"].nunique())
                self.assertEqual(np.linalg.matrix_rank(x), k)
                self.assertEqual(int(fit.result.nobs), len(data))
                np.testing.assert_array_equal(fit.sample["source_row"], np.arange(len(data)))
                np.testing.assert_allclose(fit.ordinary.params, coefficients, atol=1e-11)
                np.testing.assert_allclose(fit.result.params, fit.ordinary.params, atol=0, rtol=0)
                self.assertAlmostEqual(fit.correction_factor, correction)
                np.testing.assert_allclose(
                    fit.result.cov_params(), expected_covariance, rtol=1e-8, atol=1e-11
                )

                j = fit.result.model.exog_names.index("treated_post")
                beta = coefficients[j]
                se = np.sqrt(expected_covariance[j, j])
                t_statistic = beta / se
                p_value = 2 * stats.t.sf(abs(t_statistic), df=g - 1)
                margin = stats.t.ppf(0.975, df=g - 1) * se
                row = policy_row(fit)
                for field, expected in {
                    "coefficient": beta,
                    "std_error": se,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "ci_lower": beta - margin,
                    "ci_upper": beta + margin,
                }.items():
                    self.assertAlmostEqual(row[field], expected, places=8, msg=field)
                self.assertEqual(row["nobs"], n)
                self.assertEqual(row["n_clusters"], g)
                self.assertEqual(row["inference_df"], g - 1)
                self.assertEqual(row["n_coefficients"], k)
                checks = verify_fit(fit)
                self.assertIsInstance(checks, dict)
                json.dumps(checks, allow_nan=False)

    def test_outcome_specific_missingness_and_cluster_alignment(self):
        data = make_panel().sample(frac=1, random_state=7).copy()
        # Duplicate, unordered source labels must not be used as row positions.
        data.index = np.arange(len(data)) % 11 + 100
        data[["treatment", "post"]] = data[["treatment", "post"]].astype(float)
        wyoming_rows = np.flatnonzero(data["state_name"].eq("Wyoming"))
        other_rows = np.flatnonzero(~data["state_name"].eq("Wyoming"))[:9]

        def put(position, column, value):
            data.iloc[position, data.columns.get_loc(column)] = value

        data.loc[data["state_name"].eq("Wyoming"), "hours_worked"] = np.nan
        put(other_rows[0], "hours_worked", np.nan)
        put(other_rows[1], "ln_wages", np.nan)
        put(other_rows[2], "state_name", None)
        put(other_rows[3], "date", pd.NaT)
        put(other_rows[4], "treatment", np.nan)
        put(other_rows[5], "post", np.inf)
        put(other_rows[6], "hours_worked", np.inf)
        # These values must not introduce new employment/sample restrictions.
        put(other_rows[7], "wages", 0)
        put(other_rows[8], "hours_worked", 0)
        original = data.copy(deep=True)

        shared_exclusions = set(other_rows[2:6])
        expectations = {
            "hours_worked": (set(wyoming_rows) | shared_exclusions
                             | {other_rows[0], other_rows[6]}, 4),
            "ln_wages": (shared_exclusions | {other_rows[1]}, 5),
        }
        for outcome, (excluded, expected_g) in expectations.items():
            with self.subTest(outcome=outcome):
                fit = fit_did(data, outcome)
                expected_positions = [i for i in range(len(data)) if i not in excluded]
                np.testing.assert_array_equal(fit.sample["source_row"], expected_positions)
                np.testing.assert_array_equal(fit.sample.index, expected_positions)
                np.testing.assert_array_equal(
                    fit.sample["state_name"], data.iloc[expected_positions]["state_name"]
                )
                np.testing.assert_array_equal(
                    fit.ordinary.model.endog, data.iloc[expected_positions][outcome]
                )
                row = policy_row(fit)
                self.assertEqual(row["nobs"], len(expected_positions))
                self.assertEqual(row["n_clusters"], expected_g)
                self.assertEqual(row["inference_df"], expected_g - 1)
                self.assertEqual(row["n_coefficients"], expected_g + data["date"].nunique())
                se = row["std_error"]
                self.assertAlmostEqual(
                    row["p_value"],
                    2 * stats.t.sf(abs(row["coefficient"] / se), expected_g - 1),
                )
                verify_fit(fit)
        pd.testing.assert_frame_equal(data, original)

    def test_rejects_one_cluster(self):
        data = make_panel()
        with self.assertRaises(ValueError):
            fit_did(data.loc[data["state_name"].eq("Kansas")], "ln_wages")

    def test_rejects_unidentified_policy_coefficient(self):
        data = make_panel()
        data["treatment"] = 0
        with self.assertRaises(ValueError):
            fit_did(data, "ln_wages")


if __name__ == "__main__":
    unittest.main()
