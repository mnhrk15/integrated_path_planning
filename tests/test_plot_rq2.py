"""Tests for the RQ2 loss-surface ridge figure (synthetic npz, no real data).

Verifies that non-finite grid cells are masked on load (so they never blow up
the colour scale) and that a single surface renders without raising under the
Agg backend. Pixel content is not checked.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.plot_rq2_loss_surface import FIGS, REPO, load_surface, plot_single


def _write_npz(path, grid_loss):
    grid_sigma = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    grid_v0 = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    np.savez(path, grid_sigma=grid_sigma, grid_v0=grid_v0,
             grid_loss=grid_loss, sigma=1.2, v0=1.6,
             grid_best=np.array([1.0, 2.0]))


def test_load_surface_masks_non_finite(tmp_path):
    loss = np.random.RandomState(0).rand(6, 7) + 0.2
    loss[2, 3] = np.inf  # a degenerate cell
    loss[4, 1] = np.nan
    npz = tmp_path / "loss_surface_all.npz"
    _write_npz(npz, loss)

    surf = load_surface(npz)
    masked = surf["loss"]
    assert masked.shape == (6, 7)
    assert masked.mask[2, 3]  # inf -> masked
    assert masked.mask[4, 1]  # nan -> masked
    assert not masked.mask[0, 0]  # finite cell stays
    assert surf["sigma"] == 1.2 and surf["v0"] == 1.6
    assert tuple(surf["grid_best"]) == (1.0, 2.0)


def test_plot_single_renders_without_error(tmp_path):
    loss = np.linspace(0.4, 0.9, 42).reshape(6, 7)
    loss[0, 0] = np.inf  # one masked cell must not break drawing
    npz = tmp_path / "loss_surface_all.npz"
    _write_npz(npz, loss)

    fig, ax = plt.subplots()
    mesh = plot_single(ax, load_surface(npz), "test surface")
    assert mesh is not None  # a drawable mappable was returned
    plt.close(fig)


def test_plot_single_all_masked_is_graceful(tmp_path):
    loss = np.full((6, 7), np.inf)  # every cell degenerate
    npz = tmp_path / "loss_surface_all.npz"
    _write_npz(npz, loss)

    fig, ax = plt.subplots()
    mesh = plot_single(ax, load_surface(npz), "empty")
    assert mesh is None  # nothing drawable, but no exception
    plt.close(fig)


def test_default_output_dir_is_repo_relative():
    assert FIGS == REPO / "figs"
