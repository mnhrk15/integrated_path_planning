#!/usr/bin/env python3
"""Demo script for animated visualization.

This demonstrates the advanced animation features including:
- Animated trajectories
- Predicted pedestrian paths
- Real-time metrics
- Export to GIF/MP4
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from src.visualization import create_simple_animation


def main():
    """Run simulation and create animation."""
    logger.info("=" * 60)
    logger.info("ANIMATED VISUALIZATION DEMO")
    logger.info("=" * 60)
    
    # Load scenario
    # Define scenarios to demonstrate
    scenarios = [
        'scenarios/scenario_01.yaml',
        'scenarios/scenario_03.yaml'
    ]
    
    # Use first scenario for now or iterate
    scenario_path = Path(__file__).parent.parent / scenarios[0]
    logger.info(f"\nLoading scenario: {scenario_path.name}")
    config = load_config(str(scenario_path))
    
    # Run simulation
    logger.info("\nRunning simulation...")
    simulator = IntegratedSimulator(config)
    results = simulator.run(n_steps=150)  # Run for 15 seconds
    
    logger.info(f"✓ Simulation complete ({len(results)} frames)")
    
    # Create output directory
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create animations in both formats
    formats = [
        ('gif', 'pillow', 10),
        ('mp4', 'ffmpeg', 20)
    ]
    
    for fmt, writer, fps in formats:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Creating {fmt.upper()} animation...")
        logger.info(f"{'=' * 60}")
        
        animation_path = output_dir / f"demo_animation.{fmt}"
        
        try:
            animator = create_simple_animation(
                results=results,
                output_path=animation_path,
                show=False,  # Don't show, just save
                show_predictions=True,
                show_metrics=True,
                trail_length=30,
                fps=fps
            )
            
            file_size = animation_path.stat().st_size / (1024 * 1024)
            logger.success(f"✓ {fmt.upper()} animation saved!")
            logger.info(f"  Path: {animation_path}")
            logger.info(f"  Size: {file_size:.2f} MB")
            logger.info(f"  FPS:  {fps}")
            logger.info(f"  Frames: {len(results)}")
            
        except Exception as e:
            logger.error(f"✗ Failed to create {fmt.upper()} animation: {e}")
            
            if fmt == 'gif':
                logger.info("\nTo create GIF animations, install:")
                logger.info("  pip install pillow")
            else:
                logger.info("\nTo create MP4 animations, install:")
                logger.info("  1. ffmpeg binary:")
                logger.info("     - Ubuntu/Debian: sudo apt-get install ffmpeg")
                logger.info("     - macOS: brew install ffmpeg")
                logger.info("     - Windows: https://ffmpeg.org/download.html")
                logger.info("  2. Python package: pip install ffmpeg-python")
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("ANIMATION DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nGenerated files:")
    for f in output_dir.glob("demo_animation.*"):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
    
    logger.info("\n📝 Tips:")
    logger.info("  - Use --animate flag in run_simulation.py for automatic animation")
    logger.info("  - GIF is good for quick previews and web embedding")
    logger.info("  - MP4 provides better quality and smaller file size")
    logger.info("  - Adjust --fps to control animation speed")


if __name__ == '__main__':
    main()
