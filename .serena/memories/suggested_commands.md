# Suggested Commands
- **Setup**: `python -m venv .venv && source .venv/bin/activate`; `pip install --upgrade pip`; `pip install -r requirements.txt`; `pip install -e .`
- **Download SGAN models (required)**: `python scripts/download_sgan_models.py` (add `--pooling` to grab pooling variants) or `bash scripts/download_sgan_models.sh`.
- **Run simulation (CLI)**: `python examples/run_simulation.py --scenario scenarios/scenario_01_crossing.yaml` (add `--steps N`, `--output output/my_run`, `--log-level DEBUG`, `--animate --animation-format gif|mp4 --fps 10`).
- **Animation demo**: `python examples/demo_animation.py` (generates GIF + MP4 under configured output path).
- **Visualization in code**: see README/QUICKSTART snippets using `IntegratedSimulator` + `create_simple_animation`.
- **Sanity check**: `python scripts/check_implementation.py` (verifies deps/files/models).
- **Tests**: `pytest tests/ -v` (long-run stability test uses scenario_01, shorter total_time); optional focused runs e.g., `pytest tests/test_trajectory_predictor.py -v`.