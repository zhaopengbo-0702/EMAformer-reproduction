"""Acceptance statistics fixtures only; no training or checkpoint construction."""
from copy import deepcopy
import unittest

from build_report import metric_columns, selected_manifest, tables


class CloseoutTests(unittest.TestCase):
    def test_scope_filters_only_main288(self):
        runs = [{"id": f"m{x}", "group": "multivariate_forecasting"} for x in range(288)]
        result = selected_manifest({"runs": runs + [{"id": "branch", "group": "armor_ablation"}]})
        self.assertEqual(len(result["runs"]), 288)
        with self.assertRaises(ValueError):
            selected_manifest({"runs": runs[:-1]})

    def test_published_diff_and_missing_sd_stay_separate(self):
        row = {"mse": {"mean": .375, "sd_population": None, "sd_sample": None},
               "mae": {"mean": .391, "sd_population": None, "sd_sample": None}}
        output = metric_columns(row, {"mse": .374, "mae": .39})
        self.assertAlmostEqual(output["mse_difference"], .001)
        self.assertIsNone(output["mse_sd_population"])
        self.assertEqual(output["mse_published_reference"], .374)
        self.assertEqual(output["mse_measured_mean"], .375)

    def test_roles_same_seeds_and_multiple_protocols_rejected(self):
        row = {"dataset": "ETTh1", "horizon": 96, "checkpoint_role": "release", "n_seeds": 1,
               "seeds": [2023], "six_seed_complete": False, "protocol": "release-protocol",
               "mse": {"mean": .375, "sd_population": None, "sd_sample": None},
               "mae": {"mean": .391, "sd_population": None, "sd_sample": None}}
        best = deepcopy(row)
        best.update(checkpoint_role="best-validation", protocol="best-protocol")
        best["mse"]["mean"] = .373
        result = {"table5_by_horizon": [row, best], "table1_horizon_averages": []}
        table5, table1, comparisons = tables(result, {})
        self.assertEqual(len(comparisons), 1)
        self.assertAlmostEqual(comparisons[0]["mse_best_minus_release"], -.002)
        result["table5_by_horizon"].append(deepcopy(row))
        with self.assertRaisesRegex(ValueError, "Multiple protocols"):
            tables(result, {})


if __name__ == "__main__":
    unittest.main()
