# Contributing to DeepClean

Thank you for your interest in contributing to DeepClean — Adaptive Dual-Mode Underwater Trash Detection System.

---

## How to Contribute

### Reporting Bugs

Please open a GitHub Issue with:
- A clear title and description
- Steps to reproduce the problem
- Expected vs actual behaviour
- Your environment (OS, Python version, GPU, CUDA version)
- Relevant log output or screenshots

### Suggesting Enhancements

Open a GitHub Issue tagged `enhancement` with:
- What problem the enhancement solves
- A concrete proposed approach or design sketch
- Any relevant related work or references

### Submitting Code

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Write tests** for any new controller logic, CS changes, or evaluation scripts.
   All tests live in `tests/` and run with:
   ```bash
   pytest tests/ -v
   ```

4. **Follow code style:**
   - PEP 8 for all Python files
   - Docstrings on all public functions and classes (NumPy style)
   - Type hints on all function signatures
   - Maximum line length: 100 characters

5. **Run the full test suite** before opening a PR:
   ```bash
   pytest tests/ -v --tb=short
   python generate_synthetic_video.py
   python run_adaptive_test.py --video synthetic_underwater.mp4
   ```

6. **Open a Pull Request** against `main` with:
   - A clear description of the change
   - Reference to any related Issues
   - Confirmation that all tests pass

---

## Areas Where Contributions Are Welcome

| Area | Examples |
|---|---|
| **Controller** | New layer variants, adaptive threshold strategies, layer priority schemes |
| **Parameter Extractor** | New P1–P20 candidates, improved turbidity/blur estimators |
| **Datasets** | Additional underwater datasets, better class-mapping rules |
| **Training** | Support for new YOLO architectures (v11, RT-DETR, etc.) |
| **Evaluation** | New metrics, additional statistical tests, hardware benchmarks |
| **Visualisation** | Interactive dashboards, better HUD overlays |
| **Documentation** | Tutorials, deployment guides, hardware integration notes |

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
All contributors are expected to uphold it. Please report unacceptable behaviour to the project maintainers.

---

## Development Setup

```bash
# Clone
git clone https://github.com/your-org/DeepClean-AdaptiveDualMode.git
cd DeepClean-AdaptiveDualMode

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install with dev extras
pip install -e ".[dev,datasets]"

# Generate test video (no GPU needed)
python generate_synthetic_video.py

# Run all tests
pytest tests/ -v

# Generate all paper figures
python evaluation/generate_plots.py
```

---

## Project Structure Quick Reference

```
controller/     ← Core contribution — touch carefully, write tests for everything
datasets/       ← Download + merge utilities
training/       ← YOLO training scripts
evaluation/     ← Benchmarking, ablation, statistical tests
tests/          ← pytest unit tests (keep coverage high)
docs/           ← SVG diagrams — edit with Inkscape or any SVG editor
notebooks/      ← Colab notebook (keep in sync with run_adaptive_test.py)
```

---

## Maintainers

- Amol Bhatia — framework design, paper
- Krisvarish V. — hardware integration, experiments

Questions? Open an Issue or start a Discussion on GitHub.
