#!/usr/bin/env python3
"""Example script to run integrated path planning simulation.

This script demonstrates how to use the integrated simulator.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from src.visualization import create_simple_animation


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run integrated path planning simulation'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='scenarios/scenario_01_crossing.yaml',
        help='Path to scenario configuration file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of simulation steps (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Generate animation (GIF or MP4)'
    )
    parser.add_argument(
        '--animation-format',
        type=str,
        default='gif',
        choices=['gif', 'mp4'],
        help='Animation format (gif or mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for animation'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=args.log_level
    )
    
    # Load configuration
    logger.info(f"Loading scenario from {args.scenario}")
    config = load_config(args.scenario)
    
    # Override output path if specified
    if args.output is not None:
        config.output_path = args.output
    
    # Create simulator
    logger.info("Creating integrated simulator")
    simulator = IntegratedSimulator(config)
    
    # Run simulation
    logger.info("Starting simulation")
    results = simulator.run(n_steps=args.steps)
    
    # Save results
    logger.info("Saving results")
    simulator.save_results()
    
    # Visualize
    if config.visualization_enabled:
        logger.info("Generating visualization")
        simulator.visualize()
    
    # Generate animation if requested
    if args.animate:
        logger.info(f"Generating animation ({args.animation_format.upper()})...")
        output_dir = Path(config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ext = '.gif' if args.animation_format == 'gif' else '.mp4'
        animation_path = output_dir / f"simulation{ext}"
        
        try:
            create_simple_animation(
                results=results,
                output_path=animation_path,
                show=False,  # Don't show, just save
                show_predictions=True,
                show_metrics=True,
                fps=args.fps
            )
            logger.success(f"✓ Animation saved to {animation_path}")
        except Exception as e:
            logger.error(f"Failed to create animation: {e}")
            logger.info("Make sure you have:")
            if args.animation_format == 'gif':
                logger.info("  - pillow installed: pip install pillow")
            else:
                logger.info("  - ffmpeg installed: apt-get install ffmpeg (or brew install ffmpeg)")
                logger.info("  - ffmpeg-python installed: pip install ffmpeg-python")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total steps: {len(results)}")
    logger.info(f"Total time: {results[-1].time:.2f}s")
    logger.info(f"Final ego position: ({simulator.ego_state.x:.2f}, {simulator.ego_state.y:.2f})")
    logger.info(f"Final ego velocity: {simulator.ego_state.v:.2f} m/s")
    
    # Safety metrics
    min_distances = [r.metrics.get('min_distance', float('inf')) for r in results]
    collisions = [r.metrics.get('collision', False) for r in results]
    
    logger.info(f"Minimum distance to pedestrian: {min(min_distances):.2f}m")
    logger.info(f"Number of collisions: {sum(collisions)}")
    
    if sum(collisions) > 0:
        logger.error("⚠️  COLLISION OCCURRED!")
    else:
        logger.success("✓ No collisions")
    
    logger.info("=" * 60)
    logger.success("Simulation complete!")


if __name__ == '__main__':
    main()
