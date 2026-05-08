import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from soiltextureplot import (
    PolygonClassifier,
    TextureSystem,
    get_texture_system,
    plot_triangle_with_points,
)
from soiltextureplot.systems import list_texture_systems
from soiltextureplot.triangle import SoilTextureTriangle


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "sand": [65.0, 70.0, 35.0],
            "silt": [20.0, 24.0, 45.0],
            "clay": [15.0, 6.0, 20.0],
        }
    )


def test_imports_and_public_symbols_smoke() -> None:
    assert PolygonClassifier is not None
    assert TextureSystem is not None
    assert get_texture_system is not None
    assert plot_triangle_with_points is not None
    assert SoilTextureTriangle is not None


def test_system_registry_smoke() -> None:
    systems = list_texture_systems()
    assert "USDA" in systems
    assert "HYPRES" in systems
    assert get_texture_system("USDA").name == "USDA"


def test_classification_smoke() -> None:
    tri = SoilTextureTriangle(system_name="USDA").load_dataframe(_sample_df())
    out = tri.classify()
    assert "texture_class" in out.columns
    assert len(out) == 3
    assert out["texture_class"].notna().all()


def test_plot_smoke() -> None:
    df = _sample_df()
    system = get_texture_system("USDA")
    fig, ax = plot_triangle_with_points(df=df, system=system)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)
