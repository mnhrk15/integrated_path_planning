"""Tests for the DUT auxiliary ledger sidecar (review 1.2-3).

The DUT fidelity KS p-values used to live only in the fidelity CSVs, bypassing
the multiplicity ledger. run_rq2_dut_validation now files them as AUXILIARY
tests -- and can regenerate the sidecar from the committed CSVs verbatim
(--sidecar-from-csv), because re-running the validation at today's CLI defaults
would silently change the artifacts (they were produced at (1.156, 1.681)).
"""
import json

import numpy as np
import pandas as pd
import pytest

from examples.make_multiplicity_ledger import _is_auxiliary, assemble
from examples.run_rq2_dut_validation import (
    dut_headline_tests,
    sidecars_from_csv,
)


def _rows():
    return [
        {"group": "calibrated", "sigma": 1.156, "v0": 1.681,
         "n_encounters": 58, "ks_closest": 0.293, "p_closest": 0.0133},
        {"group": "AVEC default", "sigma": 0.7, "v0": 3.5,
         "n_encounters": 58, "ks_closest": 0.276, "p_closest": 0.0238},
        {"group": "no repulsion", "sigma": 1.0, "v0": 0.0,
         "n_encounters": 58, "ks_closest": 0.362, "p_closest": np.nan},
    ]


def test_dut_headline_tests_are_auxiliary_and_skip_nan():
    tests = dut_headline_tests(_rows(), multivehicle=True)

    assert [t["test_id"] for t in tests] == [
        "rq2.dut.multivehicle.closest_ks.calibrated",
        "rq2.dut.multivehicle.closest_ks.avec_default",
    ]  # the NaN-p row is not a hypothesis
    assert all(t["auxiliary"] is True and t["headline"] is False for t in tests)
    assert all(t["family"] == "rq2_dut_fidelity_ks_multivehicle" for t in tests)
    assert all("pseudo-replication" in t["caveat"] for t in tests)
    # The ledger must classify them as auxiliary and keep them out of overall.
    assert all(_is_auxiliary(t) for t in tests)
    canonical, auxiliary = assemble(
        tests + [{"test_id": "x", "family": "f", "p_value": 0.01}], alpha=0.05)
    assert [r["test_id"] for r in canonical] == ["x"]
    assert len(auxiliary) == 2


def test_dut_single_mode_caveat_has_no_multivehicle_clause():
    tests = dut_headline_tests(_rows(), multivehicle=False)
    assert tests[0]["family"] == "rq2_dut_fidelity_ks_single"
    assert "pseudo-replication" not in tests[0]["caveat"]


def test_sidecars_from_csv_round_trip(tmp_path):
    pd.DataFrame(_rows()).to_csv(tmp_path / "dut_fidelity_multivehicle.csv",
                                 index=False)
    wrote = sidecars_from_csv(tmp_path)

    assert [p.name for p in wrote] == ["headline_tests_dut_multivehicle.json"]
    data = json.loads(wrote[0].read_text())
    assert data["source"] == "RQ2-DUT-multivehicle"
    assert len(data["tests"]) == 2
    assert data["tests"][0]["p_value"] == pytest.approx(0.0133)

    # No CSVs at all -> nothing written (the CLI turns this into a SystemExit).
    assert sidecars_from_csv(tmp_path / "empty") == []
