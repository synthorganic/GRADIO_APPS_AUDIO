# Repository Guidelines

## Getting Started
- Use **Python 3.10+**. Create an isolated environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install runtime dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  If a package is missing, consult scripts in the repo for hints (e.g., `librosa`, `gradio`).

## Project Structure
- Top-level scripts such as `wan2audio.py` provide audio utilities.
- Tests live under `tests/`; assets for tests are in `assets/`.
- Use relative imports within the repository (`from saved_by_zero import util`).

## Coding Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Format code with [Black](https://black.readthedocs.io/) and sort imports with [isort](https://pycqa.github.io/isort/).
  ```bash
  black .
  isort .
  ```
- Include type hints and docstrings for public functions.
- Keep functions short and prefer expressive variable names.

## Testing
- Ensure all tests pass before submitting changes:
  ```bash
  pytest
  ```
- Add new tests when adding features or fixing bugs. Place them in `tests/` following the naming pattern `test_<feature>.py`.

## Documentation
- Update README files or in-line comments when behavior changes.
- Examples in notebooks (`prompt_notebook.py`) should include brief explanations and runnable code blocks.

## Git Workflow
- Keep commits focused; write descriptive messages in the imperative mood, e.g., `fix: handle missing audio file`.
- Run `git status` before committing to confirm the worktree is clean.
- Submit pull requests targeting the `main` branch.

## Support
For questions or reviews, open an issue describing the problem and proposed solution.
