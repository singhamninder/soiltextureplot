# Soil Texture Plot

`soiltextureplot` helps you classify and visualize soil samples on ternary texture triangles.

Use this package when you want to:

- classify samples with the USDA or HYPRES systems
- plot samples on a ternary soil texture diagram
- combine classification and plotting in a small Python workflow

## What You Need

- Python 3.12+
- tabular data with `clay`, `sand`, and `silt` percentages
- values that sum to 100 for each sample

## Supported Systems

- **USDA**: United States Department of Agriculture soil texture classes
- **HYPRES**: HYdraulic PRoperties of European Soils classes

## Typical Workflow

1. Install the package.
2. Load your data into a pandas DataFrame.
3. Pick a texture system.
4. Classify points and/or generate a ternary plot.

If you want the simplest starting point, go straight to the [Quickstart](quickstart.md) example.

Move to [Installation](installation.md) to get started.
