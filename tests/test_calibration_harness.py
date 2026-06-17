"""Tests for the RQ2 ego-repulsion calibration harness (synthetic data).

Covers the dataset-side alignment / encounter extraction, the pysocialforce
corrections baked into the harness (no desired-speed inflation, no stop near a
reached goal), the rollout fitter / one-step diagnostic behaviour, and an
end-to-end parameter-recovery test that exercises the grid+Nelder-Mead optimiser.
All inputs are constructed in-process so the suite never depends on the (gitignored)
real VCI files.
"""
import numpy as np
import pytest

from src.datasets.vci_loader import AgentTracks, ClipTracks
from src.datasets.vci_encounter import (
    Encounter,
    align_clip_to_grid,
    encounters_from_clips,
    encounters_from_clips_multivehicle,
    extract_encounters,
    _split_clip_per_vehicle,
)
from src.calibration import calibrate
from src.simulation.calibration_harness import (
    _cruise_speeds,
    _far_goals,
    cruise_freewalk,
    cruise_upper_quantile,
    fidelity_report,
    objective_one_step,
    objective_rollout_ade,
    simulate_encounter,
)

DT = 0.4


# --------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------- #
def make_encounter(
    T=16,
    ped_start=(0.0, -4.0),
    ped_vel=(0.0, 1.3),
    ego_start=(-6.0, 0.0),
    ego_vel=(1.5, 0.0),
    n_extra_far=0,
):
    """One ped crossing the path of an ego that drives along +x through y=0.

    With ego_vel along +x and the ped crossing +y at x=0, the ego passes near the
    ped around mid-window -> a genuine encounter. ``n_extra_far`` adds peds far
    away (never interacting) to test interaction filtering / fixed population.
    """
    times = DT * np.arange(T)
    ego_xy = np.array(ego_start)[None, :] + np.array(ego_vel)[None, :] * times[:, None]
    ego_psi = np.full(T, np.arctan2(ego_vel[1], ego_vel[0]))
    ego_vel_mag = np.full(T, float(np.hypot(*ego_vel)))

    peds = [(ped_start, ped_vel)]
    for k in range(n_extra_far):
        peds.append(((20.0 + 3 * k, -4.0), (0.0, 1.3)))
    N = len(peds)
    ped_xy = np.empty((T, N, 2))
    ped_v = np.empty((T, N, 2))
    for j, (p0, v) in enumerate(peds):
        ped_xy[:, j, :] = np.array(p0)[None, :] + np.array(v)[None, :] * times[:, None]
        ped_v[:, j, :] = np.array(v)[None, :]
    return Encounter(
        clip="synthetic",
        times=times,
        ego_xy=ego_xy,
        ego_psi=ego_psi,
        ego_vel=ego_vel_mag,
        ped_xy=ped_xy,
        ped_vel=ped_v,
        ped_ids=np.arange(N),
        dt=DT,
        min_separation=float(np.min(np.linalg.norm(ped_xy - ego_xy[:, None, :], axis=2))),
    )


def make_clip(n_veh=1, ped_offset_frames=0, fps=10.0):
    """Build a ClipTracks with one crossing ped and ``n_veh`` vehicles.

    ``ped_offset_frames`` shifts the pedestrian grid origin relative to the
    vehicle grid to exercise the alignment interpolation.
    """
    T = 12
    veh_times = DT * np.arange(T)
    veh_pos = np.empty((T, n_veh, 2))
    psi = np.empty((T, n_veh))
    vel = np.empty((T, n_veh))
    for v in range(n_veh):
        veh_pos[:, v, 0] = -6.0 + 1.5 * veh_times + 3 * v
        veh_pos[:, v, 1] = 0.0
        psi[:, v] = 0.0
        vel[:, v] = 1.5
    veh = AgentTracks(times=veh_times, ids=np.arange(n_veh),
                      positions=veh_pos, extra={"psi": psi, "vel": vel})

    ped_times = DT * (np.arange(T) + ped_offset_frames)
    ped_pos = np.empty((T, 1, 2))
    ped_pos[:, 0, 0] = 0.0
    ped_pos[:, 0, 1] = -4.0 + 1.3 * ped_times
    ped = AgentTracks(times=ped_times, ids=np.array([0]),
                      positions=ped_pos, extra={})
    return ClipTracks(clip="syn", dataset="citr", scenario="vci_front",
                      ped=ped, veh=veh, ped_path=None, veh_path=None, fps=fps)


# --------------------------------------------------------------------------- #
# alignment + encounter extraction
# --------------------------------------------------------------------------- #
def test_align_single_vehicle_onto_ped_grid():
    clip = make_clip(n_veh=1, ped_offset_frames=0)
    aligned = align_clip_to_grid(clip)
    assert aligned.times.shape[0] == clip.ped.positions.shape[0]
    assert aligned.ego_xy.shape == (aligned.times.shape[0], 2)
    # vehicle drives along +x at y=0: ego_x increases, ego_y stays 0
    assert np.all(np.diff(aligned.ego_xy[:, 0]) > 0)
    assert np.allclose(aligned.ego_xy[:, 1], 0.0)
    assert np.allclose(aligned.ego_vel, 1.5, atol=1e-6)


def test_align_offset_grid_interpolates():
    """A half-frame ped/veh grid offset must still yield finite, sane ego positions."""
    clip_off = make_clip(ped_offset_frames=0)
    # shift ped grid by half a frame so it no longer coincides with the veh grid
    clip_off.ped.times[:] = clip_off.ped.times + DT / 2
    aligned = align_clip_to_grid(clip_off)
    # The +DT/2 shift pushes exactly the last ped frame past the vehicle's support,
    # so precisely one tail frame must be NaN. Asserting == (not >=) pins the
    # no-extrapolation guarantee: with >=, code that extrapolated every frame
    # (all finite) would still pass.
    finite = np.isfinite(aligned.ego_xy).all(axis=1)
    assert finite.sum() == aligned.times.shape[0] - 1
    assert not finite[-1]  # the out-of-support tail frame is not extrapolated
    assert np.all(np.diff(aligned.ego_xy[finite, 0]) > 0)


def test_align_multi_vehicle_raises():
    with pytest.raises(ValueError, match="single ego vehicle"):
        align_clip_to_grid(make_clip(n_veh=2))


def test_align_missing_side_raises():
    clip = make_clip()
    clip.veh = None
    with pytest.raises(ValueError, match="both pedestrian and vehicle"):
        align_clip_to_grid(clip)


def test_extract_encounter_fixed_population_and_interaction():
    aligned = align_clip_to_grid(make_clip())
    encs = extract_encounters(aligned, min_sep_threshold=8.0, min_len=5)
    assert len(encs) == 1
    enc = encs[0]
    assert enc.ped_xy.shape[1] == 1  # one ped present throughout
    assert not np.any(np.isnan(enc.ped_xy))
    assert enc.min_separation < 8.0


def test_extract_drops_non_interacting_span():
    aligned = align_clip_to_grid(make_clip())
    # threshold below the actual closest approach -> no encounter qualifies
    encs = extract_encounters(aligned, min_sep_threshold=0.1, min_len=5)
    assert encs == []


def test_encounters_from_clips_skips_multivehicle():
    good = make_clip(n_veh=1)
    bad = make_clip(n_veh=2)
    encs = encounters_from_clips([good, bad], min_sep_threshold=8.0, min_len=5)
    assert len(encs) == 1  # the 2-vehicle clip is skipped, not an error


# --------------------------------------------------------------------------- #
# pysocialforce corrections (Blockers A and B)
# --------------------------------------------------------------------------- #
def test_no_ego_control_walks_straight_at_recorded_speed():
    """With v0=0 (no ego reaction) peds must keep recorded speed and direction.

    Catches the desired-speed inflation (Blocker A: would be ~1.3x too fast) and
    the stop-when-arrived freeze (Blocker B: would halt before the window ends).
    """
    enc = make_encounter(T=16, ped_vel=(0.0, 1.3))
    sim_xy = simulate_encounter(enc, sigma=0.7, v0=0.0)
    sim_speed = np.linalg.norm(np.diff(sim_xy, axis=0), axis=2) / enc.dt  # [T-1, N]
    assert np.allclose(np.median(sim_speed), 1.3, atol=0.1)  # not 1.3*1.3=1.69
    # net displacement matches a straight 1.3 m/s walk (no early stop)
    sim_net = np.linalg.norm(sim_xy[-1, 0] - sim_xy[0, 0])
    assert sim_net == pytest.approx(1.3 * enc.dt * (enc.ped_xy.shape[0] - 1), rel=0.05)


def test_ego_near_deflects_more_than_far():
    """A ped crossing close to the ego must be deflected more than one far away."""
    near = make_encounter(ped_start=(0.0, -4.0))  # crosses the ego path
    far = make_encounter(ped_start=(0.0, 40.0), ped_vel=(0.0, 1.3))  # never near ego
    dev_near = np.max(np.abs(
        simulate_encounter(near, 0.7, 3.5)[:, 0] - near.ped_xy[:, 0]))
    dev_far = np.max(np.abs(
        simulate_encounter(far, 0.7, 3.5)[:, 0] - far.ped_xy[:, 0]))
    assert dev_near > dev_far
    assert dev_far < 0.05  # the far ped is essentially unperturbed


def test_simulate_is_deterministic():
    enc = make_encounter()
    a = simulate_encounter(enc, 0.7, 3.5)
    b = simulate_encounter(enc, 0.7, 3.5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# objectives + optimiser
# --------------------------------------------------------------------------- #
def test_rollout_ade_zero_at_self_consistent_params():
    """ADE against data generated by the harness itself is ~0 at the true params."""
    base = make_encounter(T=16)
    sigma_true, v0_true = 0.7, 3.0
    goals = _far_goals(base.ped_xy, base.ped_vel)
    sim_xy = simulate_encounter(base, sigma_true, v0_true)
    # pseudo-real pins the generating goals + velocity field so the boundary
    # conditions match exactly on re-sim (=> ADE is 0 at the true params).
    pseudo = Encounter(
        clip="pseudo", times=base.times, ego_xy=base.ego_xy, ego_psi=base.ego_psi,
        ego_vel=base.ego_vel, ped_xy=sim_xy, ped_vel=base.ped_vel,
        ped_ids=base.ped_ids, dt=base.dt,
        min_separation=float(np.min(np.linalg.norm(sim_xy - base.ego_xy[:, None, :], axis=2))),
        goals=goals,
    )
    loss_true = objective_rollout_ade([pseudo], sigma_true, v0_true)
    loss_off = objective_rollout_ade([pseudo], sigma_true, 0.0)
    assert loss_true < 1e-6
    assert loss_off > loss_true


def test_calibrate_recovers_generating_params():
    """grid+Nelder-Mead recovers the (sigma, v0) that generated the pseudo data."""
    base = make_encounter(T=18)
    sigma_true, v0_true = 0.7, 3.0
    goals = _far_goals(base.ped_xy, base.ped_vel)
    sim_xy = simulate_encounter(base, sigma_true, v0_true)
    pseudo = Encounter(
        clip="pseudo", times=base.times, ego_xy=base.ego_xy, ego_psi=base.ego_psi,
        ego_vel=base.ego_vel, ped_xy=sim_xy, ped_vel=base.ped_vel,
        ped_ids=base.ped_ids, dt=base.dt, min_separation=0.0, goals=goals,
    )
    result = calibrate(
        lambda s, v: objective_rollout_ade([pseudo], s, v),
        grid_sigma=[0.3, 0.5, 0.7, 1.0, 1.5],
        grid_v0=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
    )
    assert result.grid_best == (sigma_true, v0_true)
    assert result.sigma == pytest.approx(sigma_true, abs=0.2)
    assert result.v0 == pytest.approx(v0_true, abs=0.5)
    assert result.loss < 1e-3


def test_calibrate_keeps_refinement_when_nelder_mead_hits_iter_cap():
    """A Nelder-Mead improvement must be kept even when it terminates with
    success=False (the iteration cap), not silently discarded for the on-grid
    minimum. The true (sigma, v0) is OFF the grid nodes and max_iter is tiny, so
    NM reports success=False yet still lowers the loss below every grid cell;
    gating refinement on res.success (the old bug) would return the grid point.
    """
    base = make_encounter(T=18)
    sigma_true, v0_true = 0.6, 2.7  # both strictly between grid nodes below
    goals = _far_goals(base.ped_xy, base.ped_vel)
    sim_xy = simulate_encounter(base, sigma_true, v0_true)
    pseudo = Encounter(
        clip="pseudo", times=base.times, ego_xy=base.ego_xy, ego_psi=base.ego_psi,
        ego_vel=base.ego_vel, ped_xy=sim_xy, ped_vel=base.ped_vel,
        ped_ids=base.ped_ids, dt=base.dt, min_separation=0.0, goals=goals,
    )
    result = calibrate(
        lambda s, v: objective_rollout_ade([pseudo], s, v),
        grid_sigma=[0.3, 0.5, 0.7, 1.0],
        grid_v0=[0.0, 1.0, 2.0, 3.0, 4.0],
        max_iter=3,  # forces res.success == False while still improving the loss
    )
    assert result.grid_best == (0.5, 3.0)  # nearest grid node to the truth
    assert result.refined  # the success=False improvement was kept
    assert result.loss < float(result.grid_loss.min())  # strictly below the grid
    assert result.v0 != 3.0  # moved off the grid node toward v0_true


def test_one_step_diagnostic_is_finite_and_minimal_at_zero():
    """The diagnostic runs and (as documented) does not reward repulsion here."""
    enc = make_encounter(T=16)
    encs = [enc]
    l0 = objective_one_step(encs, 0.7, 0.0)
    l_big = objective_one_step(encs, 0.7, 6.0)
    assert np.isfinite(l0)
    # Strict: a large v0 must genuinely WORSEN the instantaneous residual. A
    # non-strict <= would also pass if v0 had no effect at all (e.g. the ego
    # state never reached the force) -- exactly the regression this guards.
    assert l0 < l_big  # instantaneous force matching pulls v0 toward 0


def test_objective_returns_inf_when_no_samples():
    """An encounter with the ped always outside max_distance yields no samples."""
    enc = make_encounter(ped_start=(0.0, 40.0))
    val = objective_one_step([enc], 0.7, 3.5, max_distance=2.0)
    assert val == float("inf")


def test_rollout_ade_interaction_filter_drops_far_peds():
    enc = make_encounter(n_extra_far=2)  # 1 crossing + 2 far peds
    all_ade = objective_rollout_ade([enc], 0.7, 3.5, interaction_distance=None)
    near_ade = objective_rollout_ade([enc], 0.7, 3.5, interaction_distance=5.0)
    # both finite; filtering changes the pooled value (far peds contribute ~0 error)
    assert np.isfinite(all_ade) and np.isfinite(near_ade)
    # Strict: a no-op filter (keep every ped) would give equality and still pass a
    # non-strict >=; the far peds genuinely dilute the all-ped mean downward.
    assert near_ade > all_ade


def test_rollout_ade_excludes_trivial_frame_zero():
    """ADE must not count frame 0, whose error is structurally 0 (sim_xy[0] is
    pinned to the recorded start). Counting it would scale the loss below the
    true mean per-evolved-frame error."""
    enc = make_encounter(T=16)
    sim_xy = simulate_encounter(enc, 0.7, 3.5)
    err = np.linalg.norm(sim_xy - enc.ped_xy, axis=2)  # [T, N]
    assert np.allclose(err[0], 0.0)  # frame 0 is the recorded start by construction
    ade = objective_rollout_ade([enc], 0.7, 3.5)
    expected = float(err[1:].sum()) / err[1:].size  # frame 0 dropped
    assert ade == pytest.approx(expected)
    # and it is strictly larger than the frame-0-included mean (the old behaviour)
    diluted = float(err.sum()) / err.size
    assert ade > diluted


# --------------------------------------------------------------------------- #
# fidelity report (validation path) + optimiser guards
# --------------------------------------------------------------------------- #
def test_fidelity_report_reports_documented_keys():
    """fidelity_report runs end-to-end and returns the documented keys with a
    finite, non-negative rollout ADE and finite closest-approach stats."""
    enc = make_encounter(T=16)
    r = fidelity_report([enc], sigma=0.7, v0=3.5)
    for key in ("n_encounters", "rollout_ade", "mean_closest_sim",
                "mean_closest_real", "ks_closest", "p_closest",
                "n_onset_sim", "n_onset_real", "ks_onset", "p_onset"):
        assert key in r
    assert r["n_encounters"] == 1
    assert np.isfinite(r["rollout_ade"]) and r["rollout_ade"] >= 0.0
    assert np.isfinite(r["mean_closest_sim"]) and np.isfinite(r["mean_closest_real"])


def test_fidelity_report_empty_is_nan_not_crash():
    """No encounters -> NaN summary stats (not a divide-by-zero / empty-reduce crash)."""
    r = fidelity_report([], sigma=0.7, v0=3.5)
    assert r["n_encounters"] == 0
    assert np.isnan(r["rollout_ade"])
    assert np.isnan(r["mean_closest_sim"])


def test_calibrate_raises_when_objective_non_finite_everywhere():
    """A degenerate objective (inf on every grid cell) must raise, not silently
    return the (0,0) cell as a bogus optimum."""
    with pytest.raises(ValueError, match="non-finite loss on the entire grid"):
        calibrate(lambda s, v: float("inf"),
                  grid_sigma=[0.5, 1.0], grid_v0=[0.0, 1.0], refine=False)


# --------------------------------------------------------------------------- #
# degeneracy / NaN-robustness paths
# --------------------------------------------------------------------------- #
def test_cruise_speeds_floors_nan_and_zero():
    """A NaN-only or zero-speed column must floor to a small positive speed.

    Without the floor pysocialforce's stop-when-arrived would freeze a ped with
    desired speed 0.
    """
    ped_vel = np.zeros((5, 3, 2))
    ped_vel[:, 0, :] = np.nan          # ped 0: all NaN
    ped_vel[:, 1, :] = 0.0             # ped 1: stationary
    ped_vel[:, 2, :] = (1.0, 0.0)      # ped 2: moving at 1 m/s
    cruise = _cruise_speeds(ped_vel)
    assert cruise.shape == (3,)
    assert np.all(np.isfinite(cruise))
    assert np.all(cruise >= 1e-3)
    assert cruise[2] == pytest.approx(1.0)


def test_far_goals_finite_for_stationary_ped():
    """A ped that never moves (zero net displacement and zero velocity) still
    gets a finite goal (the +x fallback), not a NaN from a zero-norm divide."""
    ped_xy = np.zeros((4, 1, 2))       # never moves
    ped_vel = np.zeros((4, 1, 2))      # zero velocity too
    goals = _far_goals(ped_xy, ped_vel)
    assert goals.shape == (1, 2)
    assert np.all(np.isfinite(goals))


def test_extract_drops_ped_with_nan_velocity():
    """A ped present in position but with a NaN velocity sample is dropped."""
    aligned = align_clip_to_grid(make_clip())
    aligned.ped_vel[3, 0, 0] = np.nan  # poison one interior velocity sample
    encs = extract_encounters(aligned, min_sep_threshold=8.0, min_len=5)
    assert encs == []  # the only ped is excluded -> no fixed-population window


def test_extract_excludes_frames_with_nan_ego_channels():
    """Encounters are gated on ego_psi/ego_vel finiteness, not just ego_xy, so a
    frame whose ego speed/heading is NaN is excluded from the span."""
    aligned = align_clip_to_grid(make_clip())
    aligned.ego_vel[0] = np.nan  # poison the ego speed at the first frame
    encs = extract_encounters(aligned, min_sep_threshold=8.0, min_len=3)
    assert len(encs) >= 1
    for enc in encs:
        assert np.all(np.isfinite(enc.ego_vel))
        assert np.all(np.isfinite(enc.ego_psi))
        assert enc.times[0] > aligned.times[0]  # poisoned frame 0 dropped


# --------------------------------------------------------------------------- #
# multi-vehicle expansion (DUT validation, C)
# --------------------------------------------------------------------------- #
def test_split_clip_per_vehicle_count():
    """A 3-vehicle clip splits into 3 single-vehicle virtual clips sharing peds."""
    clip = make_clip(n_veh=3)
    subs = _split_clip_per_vehicle(clip)
    assert len(subs) == 3
    for sub in subs:
        assert sub.veh.positions.shape[1] == 1  # downstream single-vehicle assert holds
        assert sub.ped is clip.ped  # pedestrians shared, not copied
        assert sub.clip != clip.clip  # stem disambiguated (#v{id})


def test_split_single_vehicle_passthrough():
    """A single-vehicle clip is returned unchanged (identity), reproducing CITR."""
    clip = make_clip(n_veh=1)
    subs = _split_clip_per_vehicle(clip)
    assert subs == [clip]
    assert subs[0] is clip


def test_multivehicle_equals_single_for_one_vehicle_clip():
    """On a single-vehicle clip the multi-vehicle path is a superset == the original."""
    clip = make_clip(n_veh=1)
    a = encounters_from_clips([clip], min_sep_threshold=8.0, min_len=5)
    b = encounters_from_clips_multivehicle([clip], min_sep_threshold=8.0, min_len=5)
    assert len(a) == len(b) == 1


def test_multivehicle_extracts_from_two_vehicle_clip():
    """A two-vehicle clip yields encounters via expansion but is skipped by the
    single-vehicle extractor."""
    clip = make_clip(n_veh=2)
    single = encounters_from_clips([clip], min_sep_threshold=8.0, min_len=5)
    multi = encounters_from_clips_multivehicle([clip], min_sep_threshold=8.0, min_len=5)
    assert single == []  # multi-vehicle clip skipped by the legacy extractor
    assert len(multi) >= 1  # at least the lead vehicle crosses the ped


# --------------------------------------------------------------------------- #
# cruise-speed estimators + cruise_fn injection (RQ2 cruise-bias diagnostic, D)
# --------------------------------------------------------------------------- #
def _speed_varying_encounter(near_speed=0.5, far_speed=1.5, T=10):
    """One ped fixed at the origin; ego far (frames 0..T/2) then near (T/2..T).

    Speed is set INDEPENDENTLY of position so the estimators (which read velocity
    for speed and positions for ego distance) can be probed in isolation: the ped
    walks ``far_speed`` while the ego is far and ``near_speed`` while it is near.
    """
    half = T // 2
    times = DT * np.arange(T)
    ped_xy = np.zeros((T, 1, 2))  # ped pinned at origin
    ego_xy = np.zeros((T, 2))
    ego_xy[:half, 0] = 100.0  # far (dist 100 > 8) for the first half
    ego_xy[half:, 0] = 0.0    # near (dist 0 < 8) for the second half
    ped_vel = np.zeros((T, 1, 2))
    ped_vel[:half, 0, 1] = far_speed
    ped_vel[half:, 0, 1] = near_speed
    return Encounter(
        clip="speedvar", times=times, ego_xy=ego_xy, ego_psi=np.zeros(T),
        ego_vel=np.ones(T), ped_xy=ped_xy, ped_vel=ped_vel, ped_ids=np.array([0]),
        dt=DT, min_separation=0.0,
    )


def test_cruise_fn_injection_default_matches_baseline():
    """cruise_fn=None and an explicit baseline estimator give bit-identical rollouts."""
    enc = make_encounter()
    a = simulate_encounter(enc, 0.7, 3.5)
    b = simulate_encounter(enc, 0.7, 3.5, cruise_fn=lambda e: _cruise_speeds(e.ped_vel))
    assert np.array_equal(a, b)


def test_cruise_freewalk_excludes_near_frames():
    """Free-walking cruise uses only the far (fast) frames, beating the all-frame median."""
    enc = _speed_varying_encounter(near_speed=0.5, far_speed=1.5)
    baseline = _cruise_speeds(enc.ped_vel)[0]      # median over all frames -> ~1.0
    free = cruise_freewalk(enc, ego_distance_threshold=8.0, quantile=0.5)[0]  # far frames -> 1.5
    assert free > baseline
    assert np.isclose(free, 1.5, atol=1e-6)


def test_cruise_freewalk_fallback_when_all_near():
    """A ped that is never free-walking falls back to the all-frame median (finite, floored)."""
    enc = _speed_varying_encounter(near_speed=1.0, far_speed=1.0)
    enc.ego_xy[:] = 0.0  # ego near at every frame -> no free-walking sample
    free = cruise_freewalk(enc, ego_distance_threshold=8.0)[0]
    assert np.isfinite(free) and free >= 1e-3
    assert np.isclose(free, 1.0, atol=1e-6)  # fallback median of constant 1.0


def test_cruise_upper_quantile_above_median():
    """The 85th-percentile estimator sits above the median for a slowdown-skewed ped."""
    enc = _speed_varying_encounter(near_speed=0.5, far_speed=1.5)
    median = _cruise_speeds(enc.ped_vel)[0]
    upper = cruise_upper_quantile(enc, quantile=0.85)[0]
    assert upper >= median
