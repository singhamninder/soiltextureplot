# API Overview

This page covers the main public API for package users.

## Recommended for Most Users: `SoilTextureTriangle`

```python
from soiltextureplot.triangle import SoilTextureTriangle
```

Use this class when you want a simple, notebook-friendly workflow.

Example:

```python
import pandas as pd
from soiltextureplot.triangle import SoilTextureTriangle

example_df = pd.DataFrame(
    {
        "sample_id": ["S1", "S2", "S3"],
        "sand": [75, 65, 45],
        "silt": [15, 20, 35],
        "clay": [10, 15, 20],
        "BD": [1.35, 1.42, 1.20],
    }
)

tri = SoilTextureTriangle(system_name="USDA")
tri.load_dataframe(example_df)
classified = tri.classify()
fig, ax = tri.plot(size_by="BD", cmap="viridis")
```

Key methods:

- `load_csv(...)`: load from CSV and normalize texture column names
- `load_dataframe(...)`: load from pandas DataFrame
- `classify()`: add `texture_class` values to loaded data
- `plot(...)`: create the ternary plot figure

## Advanced API (Lower-Level Building Blocks)

Use these when you want full control over classification and plotting internals.

### `get_texture_system(system_name)`

Returns a `TextureSystem` definition used for classification and plotting.

- Accepted values: `"USDA"`, `"HYPRES"`
- Raises `ValueError` for unknown system names

### `PolygonClassifier`

Classifies points into texture classes using polygon inclusion.

```python
from soiltextureplot import PolygonClassifier, get_texture_system

system = get_texture_system("USDA")
classifier = PolygonClassifier.from_system(system)

labels = classifier.classify_points(
    clay=df["clay"].to_numpy(),
    sand=df["sand"].to_numpy(),
    silt=df["silt"].to_numpy(),
)
```

If a point is outside known polygons, the classifier returns `"Unknown"` for that point.

### `plot_triangle_with_points(...)`

Creates a ternary diagram with texture polygons and sample points.

Expected DataFrame columns:

- required: `clay`, `sand`, `silt`
- optional: `sample_id` (for labels), plus any `size_by` column

Common options:

- `show_labels=True` to draw sample labels
- `size_by`, `size_min`, `size_max` to scale point size
- `cmap` to control class polygon colors
- `color_points` to control point colors

Example:

```python
from soiltextureplot import get_texture_system, plot_triangle_with_points

system = get_texture_system("USDA")
fig, ax = plot_triangle_with_points(df=df, system=system, show_labels=True)
```

Advanced users can also import internal modules from `soiltextureplot.*` as needed.
