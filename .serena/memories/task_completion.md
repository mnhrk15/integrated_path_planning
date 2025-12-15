# Task Completion Checklist
- Run automated tests: `pytest tests/ -v` (or targeted modules) after changes; ensure scenario files/models available if tests depend on them.
- If modifying simulation/prediction paths, confirm `sgan_model_path` points to a downloaded checkpoint (no fallback) and optional dependencies (pysocialforce, pillow/ffmpeg) are noted if required for the change.
- When touching CLI/examples, do a smoke run: `python examples/run_simulation.py --scenario scenarios/scenario_01_crossing.yaml` (optionally `--steps 20` for quick check) to verify logging/output/visualization paths.
- For visualization changes, optionally run `python examples/demo_animation.py` or enable `--animate` to ensure GIF/MP4 export still works (requires pillow or ffmpeg/ffmpeg-python).
- Update docs/README/CHANGELOG snippets if behavior, flags, or config fields change.
- Keep log level defaults and device handling intact; avoid breaking Agg backend fallback for headless runs.