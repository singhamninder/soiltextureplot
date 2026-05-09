# Soil Texture Plot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soiltextureplot.streamlit.app/)

`soiltextureplot` is a Python package for soil texture classification and visualization with ternary diagrams.

## Features

- **Ternary Plotting**: Visualize soil texture data on interactive ternary diagrams
- **Multiple Classification Systems**: Support for USDA and HYPRES soil texture classification systems
- **Interactive Web App**: Streamlit-based application for easy data upload and visualization
- **Flexible Data Input**: Support for CSV files with customizable column mapping
- **Point Classification**: Automatic classification of soil samples into texture classes
- **Customizable Visualization**: Control point sizes, colors, and labels

## Install

Install with `uv`:

```bash
uv add soiltextureplot
```

Install with `pip`:

```bash
pip install soiltextureplot
```

Optional app extras:

```bash
pip install "soiltextureplot[app]"
```

## Quick Usage

```python
import pandas as pd
from soiltextureplot.triangle import SoilTextureTriangle

example_df = pd.DataFrame(
    {
        "sample_id": ["S1", "S2", "S3", "S4"],
        "sand": [75, 65, 45, 35],
        "silt": [15, 20, 35, 40],
        "clay": [10, 15, 20, 25],
        "BD": [1.35, 1.42, 1.20, 1.18],
    }
)

tri = SoilTextureTriangle(system_name="USDA")
tri.load_dataframe(example_df)
classified = tri.classify()
fig, ax = tri.plot(size_by="BD", cmap="viridis")
```

## Documentation

Full package docs are available on GitHub Pages:

- [https://singhamninder.github.io/soiltextureplot/](https://singhamninder.github.io/soiltextureplot/)

## Data Format

Your CSV file should contain soil texture data with percentages of sand, silt, and clay. The percentages should sum to 100% for each sample.

Example data format:

```csv
sample_id,sand,silt,clay
S1,65,20,15
S2,70,24,6
S3,75,21,4
```

## Supported Classification Systems

### USDA (United States Department of Agriculture)

The standard USDA soil texture classification system with 12 texture classes.

### HYPRES (HYdraulic PRoperties of European Soils)

A European framework for classifying soils based on their hydrologic properties.

## Web Application

The Streamlit web app allows you to:

- Upload CSV files with soil texture data
- Map columns to sand, silt, and clay percentages
- Visualize data on interactive texture triangles
- Customize plot appearance

## Notebook Workflows (Pilot)

During the Marimo pilot, both notebook paths are supported:

- Jupyter notebook remains available at `texture_plot.ipynb`
- Marimo notebook copy is available at `notebooks/texture_plot_marimo.py`
- Keep existing `nbqa` checks for `.ipynb` files during this phase

Run the Marimo notebook:

```bash
uv run marimo edit notebooks/texture_plot_marimo.py
```

Run the existing Jupyter notebook:

```bash
uv run jupyter notebook texture_plot.ipynb
```

## Development

```bash
# Setup
uv sync --group dev

# Install git pre-commit hook
uv run pre-commit install

# Run all pre-commit checks on the repository
uv run pre-commit run --all-files

# Run one hook manually (example: ruff)
uv run pre-commit run ruff --all-files

# CI-style check-only quality commands (no auto-fixes)
uv run ruff format --check .
uv run ruff check .
uv run ty check src app

# Run tests
uv run pytest

# Build
uv build

# Publish (after updating version in pyproject.toml)
uv publish
```

## Release Automation (GitHub Actions)

Publishing is automated via tag-triggered workflows:

- `v*rc*` tags publish to TestPyPI
- `v*.*.*` tags are evaluated for PyPI; tags containing `rc` are skipped by workflow condition

Both workflows verify that the git tag version matches `uv version --short` before publishing.
Both workflows also run isolated wheel and source-distribution smoke tests (`import soiltextureplot`) before upload.

For maintainers, full trusted-publishing and release steps are documented in [`docs/releasing.md`](docs/releasing.md).

## Acknowledgments

- Built using [mpltern](https://github.com/yuzie007/mpltern) for ternary plotting
- Inspired by soil science classification standards
- Streamlit for the web interface