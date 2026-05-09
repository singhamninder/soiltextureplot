# Soil Texture Plot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soiltextureplot.streamlit.app/)

This repository contains codebase for soil texture classification and visualization using ternary diagrams. It provides tools to plot soil texture data on texture triangles and classify soil samples according to different classification systems.

## Features

- **Ternary Plotting**: Visualize soil texture data on interactive ternary diagrams
- **Multiple Classification Systems**: Support for USDA and HYPRES soil texture classification systems
- **Interactive Web App**: Streamlit-based application for easy data upload and visualization
- **Flexible Data Input**: Support for CSV files with customizable column mapping
- **Point Classification**: Automatic classification of soil samples into texture classes
- **Customizable Visualization**: Control point sizes, colors, and labels

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- mpltern

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

### Trusted publishing prerequisites

Before pushing release tags, configure trusted publishing:

1. In GitHub, create environments named `pypi` and `testpypi` under repository settings.
2. In PyPI project settings, add a trusted publisher matching this repository and the `publish-pypi.yml` workflow.
3. In TestPyPI project settings, add a trusted publisher matching this repository and the `publish-testpypi.yml` workflow.

### Prerelease to TestPyPI

```bash
# Example: bump to prerelease version
uv version 0.1.2rc1
git add pyproject.toml uv.lock
git commit -m "Bump version to 0.1.2rc1"
git push origin dev

# After merge to main, create and push prerelease tag from main
git checkout main
git pull
git tag -a v0.1.2rc1 -m "Release v0.1.2rc1"
git push origin v0.1.2rc1
```

Smoke test from TestPyPI:

```bash
uv run --with "soiltextureplot==0.1.2rc1" --no-project -- python -c "import soiltextureplot"
```

### Stable release to PyPI

```bash
# Example: bump stable version
uv version --bump patch
git add pyproject.toml uv.lock
git commit -m "Bump version"
git push origin dev

# After merge to main, create and push stable tag from main
git checkout main
git pull
git tag -a v0.1.2 -m "Release v0.1.2"
git push origin v0.1.2
```

Smoke test from PyPI:

```bash
uv run --with soiltextureplot --no-project -- python -c "import soiltextureplot"
```

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


## Web Application

The web app allows you to:
- Upload CSV files with soil texture data
- Map columns to sand, silt, and clay percentages
- Visualize data on interactive texture triangles
- Customize plot appearance

## Supported Classification Systems

### USDA (United States Department of Agriculture)
The standard USDA soil texture classification system with 12 texture classes.

### HYPRES (HYdraulic PRoperties of European Soils)
A European framework for classifying soils based on their hydrologic properties.

## Data Format

Your CSV file should contain soil texture data with percentages of sand, silt, and clay. The percentages should sum to 100% for each sample.

Example data format:

```csv
sample_id,sand,silt,clay
S1,65,20,15
S2,70,24,6
S3,75,21,4
```

## Acknowledgments

- Built using [mpltern](https://github.com/yuzie007/mpltern) for ternary plotting
- Inspired by soil science classification standards
- Streamlit for the web interface