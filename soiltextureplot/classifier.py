from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib.path import Path

from .systems import TextureSystem
from .utils import ternary_to_cartesian


@dataclass
class PolygonClassifier:
    system: TextureSystem
    _paths: Dict[str, Path]
    _class_order: List[str]

    @classmethod
    def from_system(cls, system: TextureSystem) -> "PolygonClassifier":
        paths: Dict[str, Path] = {}

        for name, vertices in system.polygons.items():
            verts = np.array(vertices, dtype=float)  # shape (N, 3) (clay, sand, silt)
            clay, sand, silt = verts.T
            xy = ternary_to_cartesian(clay, sand, silt)
            # Ensure polygon is closed
            if not np.allclose(xy[0], xy[-1]):
                xy = np.vstack([xy, xy[0]])

            paths[name] = Path(xy)

        class_order = list(system.polygons.keys())
        return cls(system=system, _paths=paths, _class_order=class_order)

    def classify_points(self, clay, sand, silt) -> np.ndarray:
        """
        Classify many points at once.

        Returns: array of class names (dtype=object), 'Unknown' if no polygon contains the point.
        """
        xy = ternary_to_cartesian(clay, sand, silt)
        n = xy.shape[0]
        result = np.full(n, "Unknown", dtype=object)

        # Vectorized point-in-polygon: test all points against each Path
        for class_name in self._class_order:
            path = self._paths[class_name]
            inside = path.contains_points(xy)
            # Only overwrite where still Unknown
            mask = inside & (result == "Unknown")
            result[mask] = class_name

        return result
