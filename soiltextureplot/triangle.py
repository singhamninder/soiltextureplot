import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .systems import get_texture_system, TextureSystem
from .classifier import PolygonClassifier
from . import plotting


@dataclass
class SoilTextureTriangle:
    system_name: str = "USDA"
    df: Optional[pd.DataFrame] = field(default=None, repr=False)

    def __post_init__(self):
        self.system: TextureSystem = get_texture_system(self.system_name)
        self._classifier = PolygonClassifier.from_system(self.system)

    # data loading
    def load_csv(
        self,
        path: str | Path,
        sand_col: str = "sand",
        silt_col: str = "silt",
        clay_col: str = "clay",
    ) -> "SoilTextureTriangle":
        df = pd.read_csv(path)
        return self.load_dataframe(df, sand_col, silt_col, clay_col)

    def load_dataframe(
        self,
        df: pd.DataFrame,
        sand_col: str = "sand",
        silt_col: str = "silt",
        clay_col: str = "clay",
    ) -> "SoilTextureTriangle":
        # normalize column names internally
        self.df = df.rename(
            columns={sand_col: "sand", silt_col: "silt", clay_col: "clay"}
        )
        return self

    # classification
    def classify(self) -> pd.DataFrame:
        """
        Add a 'texture_class' column based on polygons for the selected system.
        For now this is a stub; later you implement point-in-polygon here.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv or load_dataframe first.")

        clay = self.df["clay"].to_numpy()
        sand = self.df["sand"].to_numpy()
        silt = self.df["silt"].to_numpy()

        classes = self._classifier.classify_points(clay, sand, silt)
        self.df["texture_class"] = classes
        return self.df

    # plotting
    def plot(
        self,
        size_by: Optional[str] = None,
        size_min: float = 40,
        size_max: float = 160,
        show_labels: bool = True,
        cmap: str = None,
        color_points: str = "black",
    ):
        """
        Plot current data on the soil texture triangle using mpltern.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv or load_dataframe first.")

        return plotting.plot_triangle_with_points(
            df=self.df,
            system=self.system,
            size_by=size_by,
            size_min=size_min,
            size_max=size_max,
            show_labels=show_labels,
            cmap=cmap,
            color_points=color_points,
        )
