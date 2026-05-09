# Quickstart

This is the fastest way to create your first USDA texture plot with `SoilTextureTriangle`.

```python
import pandas as pd

from soiltextureplot.triangle import SoilTextureTriangle

# Minimal example data (sand + silt + clay should sum to 100)
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

`classified` is a DataFrame copy with an added `texture_class` column.

## Switching to HYPRES

Change only the system name:

```python
tri = SoilTextureTriangle(system_name="HYPRES")
```

Then load the same DataFrame and call `plot()` again.
