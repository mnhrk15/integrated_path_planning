"""
Tests for animation functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.state import EgoState, PedestrianState


@pytest.fixture
def mock_results():
    """Create mock simulation results"""
    results = []
    for i in range(10):
        result = Mock()
        result.time = i * 0.1
        result.ego_state = EgoState(
            x=i * 1.0,
            y=0.0,
            yaw=0.0,
            v=5.0,
            a=0.0
        )
        result.ped_states = [
            PedestrianState(
                x=10.0 - i * 0.5,
                y=2.0,
                vx=-0.5,
                vy=0.0,
                gx=0.0,
                gy=2.0
            )
        ]
        result.predicted_trajectories = [
            np.array([[10.0 - i * 0.5 - j * 0.5, 2.0] for j in range(12)])
        ]
        result.metrics = {
            'min_distance': 5.0 - i * 0.3,
            'collision_risk': 0.1 * i
        }
        results.append(result)
    return results


def test_animator_import():
    """Test that animator module can be imported"""
    try:
        from src.visualization.animator import SimulationAnimator, create_simple_animation
        assert SimulationAnimator is not None
        assert create_simple_animation is not None
    except ImportError as e:
        pytest.skip(f"Animation dependencies not available: {e}")


def test_animator_initialization(mock_results):
    """Test animator initialization"""
    try:
        from src.visualization.animator import SimulationAnimator
        
        animator = SimulationAnimator(
            results=mock_results,
            figsize=(12, 8),
            dpi=80,
            interval=100
        )
        assert animator.results == mock_results
        assert animator.figsize == (12, 8)
        assert animator.dpi == 80
        assert animator.interval == 100
    except ImportError:
        pytest.skip("matplotlib not available")


def test_create_simple_animation_parameters(mock_results, tmp_path):
    """Test that create_simple_animation accepts correct parameters"""
    try:
        from src.visualization.animator import create_simple_animation
        
        output_path = tmp_path / "test.gif"
        
        # Test parameter validation (don't actually create animation)
        # This tests the function signature
        import inspect
        sig = inspect.signature(create_simple_animation)
        expected_params = ['results', 'output_path', 'show', 'show_predictions', 
                          'show_metrics', 'trail_length', 'fps']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Missing parameter: {param}"
            
    except ImportError:
        pytest.skip("Dependencies not available")


def test_animator_empty_results():
    """Test animator handles empty results gracefully"""
    try:
        from src.visualization.animator import SimulationAnimator
        
        empty_results = []
        
        with pytest.raises((ValueError, AssertionError)):
            SimulationAnimator(empty_results)
    except ImportError:
        pytest.skip("matplotlib not available")


def _footprint_test_animator(results):
    from src.visualization.animator import SimulationAnimator
    import matplotlib.pyplot as plt
    animator = SimulationAnimator(results=results)
    fig, ax = plt.subplots()
    animator.ax_main = ax
    from matplotlib.patches import Polygon
    artists = {
        'ego': Polygon([[0, 0], [0, 1], [1, 1], [1, 0]]),
        'ego_trail': ax.plot([], [])[0],
        'footprint': [],
    }
    ax.add_patch(artists['ego'])
    return animator, artists, fig


def test_vehicle_dimensions_come_from_parameters(mock_results):
    """The drawn body must use the configured dimensions, not hardcoded 4.5x2.0."""
    from src.visualization.animator import SimulationAnimator
    animator = SimulationAnimator(
        results=mock_results, vehicle_length=3.9, vehicle_width=1.8)
    assert animator.vehicle_length == 3.9
    assert animator.vehicle_width == 1.8


def test_footprint_overlay_circle_mode():
    """circle mode: one dashed circle of ego_radius at the vehicle centre —
    the honest overlay of what collision checking actually sees."""
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    result = SimpleNamespace(
        ego_state=SimpleNamespace(x=2.0, y=1.0, yaw=0.0),
        footprint=None, ego_radius=1.0)
    animator, artists, fig = _footprint_test_animator([result])
    animator._update_ego(artists, result, frame=0, trail_length=5)
    assert len(artists['footprint']) == 1
    circle = artists['footprint'][0]
    assert circle.get_radius() == pytest.approx(1.0)
    assert tuple(circle.get_center()) == pytest.approx((2.0, 1.0))
    plt.close(fig)


def test_footprint_overlay_multi_circle_mode():
    """multi_circle mode: one circle per footprint circle, oriented by yaw."""
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from src.core.footprint import EgoFootprint
    footprint = EgoFootprint.multi_circle(4.5, 2.0, 3)
    result = SimpleNamespace(
        ego_state=SimpleNamespace(x=0.0, y=0.0, yaw=0.0),
        footprint=footprint, ego_radius=1.0)
    animator, artists, fig = _footprint_test_animator([result])
    animator._update_ego(artists, result, frame=0, trail_length=5)
    assert len(artists['footprint']) == 3
    radii = [c.get_radius() for c in artists['footprint']]
    assert radii == pytest.approx([footprint.radius] * 3)
    xs = sorted(c.get_center()[0] for c in artists['footprint'])
    assert xs == pytest.approx(sorted(footprint.offsets))
    plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
