# Style and Conventions
- **Language/Version**: Python 3.8+; packaged via setuptools; UTF-8 files.
- **Typing/Structures**: Heavy use of `dataclass` models (`SimulationConfig`, `EgoVehicleState`, `PedestrianState`, `SimulationResult`, `FrenetPath`); numpy arrays for math-heavy data; methods often return arrays or dataclasses. Type hints used throughout.
- **Logging**: `loguru` for structured logging; logger configured in scripts with level/format; warn on fallbacks (missing pysocialforce/prediction errors).
- **Docstrings/Comments**: Module/class/function docstrings summarizing behavior and args; inline comments minimal and focused on non-obvious logic.
- **Naming**: Snake_case for functions/vars; PascalCase for classes; constants uppercase.
- **Testing**: `pytest` with fixtures/mocking; tests check shapes, fallbacks, and stability. Legacy simple dataclasses in `src/core/state.py` used by some tests.
- **Formatting/Linting**: `black`, `flake8`, `mypy` listed in requirements/extras but not enforced via config; adhere to PEP8 where possible.
- **Numerics**: Uses numpy for geometry, PyTorch for SGAN; be mindful of device placement (`torch.device`), array shapes (obs/pred dims), and safety radii when computing metrics/collisions.
- **Visualization**: Matplotlib; animator forces Agg backend if none set to avoid GUI issues; keep figures DPI/size configurable.