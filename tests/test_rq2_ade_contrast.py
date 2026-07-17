"""Tests for examples/make_rq2_ade_contrast.py (thesis review M3).

The ch06 claim "removing the ego repulsion increases the held-out ADE by ~5%"
gets its arm-contrast test here (per-arm-vs-real significance does not imply a
significant arm difference). The regression pins are against the COMMITTED
outputs/rq2_evaluation/folds_loco.csv, so a silent change to either the fold
data or the paired-test convention fails loudly.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from examples.make_rq2_ade_contrast import ade_contrast_tests

REPO = Path(__file__).parent.parent
FOLDS_CSV = REPO / "outputs" / "rq2_evaluation" / "folds_loco.csv"


def _folds(calib, norep, protocol="loco"):
    return pd.DataFrame({
        "protocol": [protocol] * len(calib),
        "test_ade": calib,
        "base_norepulsion_test_ade": norep,
    })


class TestSynthetic:
    def test_direction_is_norep_minus_calib(self):
        """d = norep - calib: a uniformly worse norep arm wins every fold."""
        tests = ade_contrast_tests(_folds([0.5] * 6, [0.7] * 6))
        sign = tests[0]
        assert sign["test_id"] == "rq2.loco.ade_sign.no_repulsion"
        assert sign["n_norep_gt_calib"] == 6
        assert sign["statistic"] == 6.0
        assert sign["mean_gap_ade"] == pytest.approx(0.2)
        assert sign["p_value"] == pytest.approx(2 * 0.5 ** 6)

    def test_zero_differences_drop_from_sign_test(self):
        """_paired_stats convention: zero diffs carry no direction."""
        tests = ade_contrast_tests(
            _folds([0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.5, 0.5]))
        sign = tests[0]
        assert sign["n_pairs"] == 4          # pairs kept
        assert sign["n_norep_gt_calib"] == 2
        assert sign["p_value"] == pytest.approx(2 * 0.5 ** 2)  # n_eff = 2

    def test_two_member_family_all_auxiliary(self):
        tests = ade_contrast_tests(_folds([0.5] * 6, [0.7] * 6))
        assert len(tests) == 2
        assert {t["family"] for t in tests} == {"rq2_ade_contrast_loco"}
        assert all(t["auxiliary"] is True for t in tests)
        assert all(t["headline"] is False for t in tests)
        assert all(t["sidedness"] == "two-sided" for t in tests)
        assert tests[1]["test_id"] == "rq2.loco.ade_wilcoxon.no_repulsion"

    def test_mixed_protocol_folds_refused(self):
        df = _folds([0.5] * 4, [0.7] * 4)
        df.loc[2, "protocol"] = "loso"
        with pytest.raises(SystemExit):
            ade_contrast_tests(df)

    def test_missing_required_column_refused(self):
        df = _folds([0.5] * 4, [0.7] * 4).drop(columns=["test_ade"])
        with pytest.raises(SystemExit, match="missing required column"):
            ade_contrast_tests(df)

    def test_all_zero_differences_serialise_as_null(self):
        """Degenerate folds must yield strict-JSON null, not a NaN token."""
        tests = ade_contrast_tests(_folds([0.5] * 4, [0.5] * 4))
        assert tests[0]["p_value"] is None      # sign: n_eff = 0
        assert tests[1]["p_value"] is None      # wilcoxon: all-zero d
        json.dumps(tests)  # must not emit bare NaN


class TestCommittedFoldsRegression:
    """Pins against the committed LOCO folds (the thesis-quoted numbers)."""

    @pytest.fixture(scope="class")
    def tests(self):
        return ade_contrast_tests(pd.read_csv(FOLDS_CSV))

    def test_sign_test_pin(self, tests):
        sign = tests[0]
        assert sign["n_pairs"] == 26
        assert sign["n_norep_gt_calib"] == 21
        assert sign["n_norep_lt_calib"] == 5    # no zero-diff folds: 21+5=26
        assert sign["p_value"] == pytest.approx(2.4939e-3, rel=1e-3)

    def test_wilcoxon_pin(self, tests):
        wil = tests[1]
        assert wil["statistic"] == pytest.approx(40.0)
        assert wil["p_value"] == pytest.approx(2.4801e-4, rel=1e-3)

    def test_mean_gap_matches_summary(self, tests):
        # summary_loco.txt: calibrated 0.640 vs no-repulsion 0.672
        assert tests[0]["mean_gap_ade"] == pytest.approx(0.0317, abs=5e-4)

    def test_deterministic_serialisation(self, tests):
        again = ade_contrast_tests(pd.read_csv(FOLDS_CSV))
        assert json.dumps(tests, sort_keys=True) \
            == json.dumps(again, sort_keys=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
