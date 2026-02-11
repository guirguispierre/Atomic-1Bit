# Contributing to Atomic-1Bit

Thanks for your interest in contributing. This project benefits from bug reports, feature ideas, documentation improvements, and code contributions.

## Reporting Issues

Open a [GitHub Issue](https://github.com/guirguispierre/Atomic-1Bit/issues) with:

- A clear title describing the problem
- Steps to reproduce (commands, config used, OS/hardware)
- Expected vs. actual behavior
- Python/compiler version and any error output

## Submitting Pull Requests

1. **Fork** the repository and create a branch from `main`
2. **Make your changes** -- keep commits focused and descriptive
3. **Run the test suite** before submitting:
   ```bash
   pytest tests/ -v
   ```
4. **Verify kernel parity** if you changed any math or kernel code:
   ```bash
   python3 atomic_1bit/python/inference.py
   ```
5. **Open a PR** with a clear description of what changed and why

### PR Guidelines

- One logical change per PR. If you're fixing a bug and adding a feature, split them.
- Include test coverage for new functionality where practical.
- Don't break existing tests. If a test needs updating, explain why in the PR.

## Code Style

This project uses pre-commit hooks for formatting:

- **Black** for Python formatting
- **isort** for import sorting
- **flake8** for linting

Install the hooks:

```bash
pip install pre-commit
pre-commit install
```

They'll run automatically on each commit. To run manually:

```bash
pre-commit run --all-files
```

### General Conventions

- Use clear variable names over comments. Comment the *why*, not the *what*.
- Keep functions focused -- one function, one job.
- Use type hints for public APIs.
- Follow existing patterns in the codebase rather than introducing new conventions.

## Testing

Tests live in `tests/` and use pytest. Run them with:

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_bitlinear.py -v

# With coverage (if installed)
pytest tests/ -v --cov=atomic_1bit
```

Key test areas:
- `test_bitlinear.py` -- BitLinear layer correctness
- `test_kernel_parity.py` -- C++/Python output matching
- `test_transformer.py` -- Full model forward pass
- `test_export.py` -- Binary export pipeline
- `test_thermal.py` -- Thermal safety monitoring

## Project Structure

If you're not sure where to make a change:

| Area | Directory | Description |
|:---|:---|:---|
| Model architecture | `atomic_1bit/model/` | Transformer, attention, gist |
| Core layers | `atomic_1bit/nn/` | BitLinear, quantization |
| Training | `atomic_1bit/training/` | Training scripts and data loading |
| Evaluation | `atomic_1bit/evaluation/` | Quality metrics |
| C++ runtime | `embedded/`, `atomic_1bit/core/` | Bare-metal inference |
| Utilities | `atomic_1bit/utils/` | Export, thermal, gist generation |
| Tests | `tests/` | pytest test suite |
| Configs | `configs/` | YAML model presets |

## Questions?

Open a [Discussion](https://github.com/guirguispierre/Atomic-1Bit/discussions) for questions about architecture, design decisions, or how to get started with a contribution.
