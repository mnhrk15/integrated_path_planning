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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
