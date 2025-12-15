"""Long-run stability test for simulation."""

import numpy as np
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator


def test_long_run_no_nan_or_inf(tmp_path):
    # Use scenario_01 with shorter dt/steps to limit runtime
    config = load_config('scenarios/scenario_01_crossing.yaml')
    config.total_time = 15.0  # reduce time for test speed
    config.output_path = tmp_path / "out"

    sim = IntegratedSimulator(config)
    results = sim.run()

    xs = np.array([r.ego_state.x for r in results])
    ys = np.array([r.ego_state.y for r in results])
    assert not np.isnan(xs).any()
    assert not np.isnan(ys).any()
    assert np.isfinite(xs).all()
    assert np.isfinite(ys).all()
