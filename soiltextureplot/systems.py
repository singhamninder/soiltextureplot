from dataclasses import dataclass
from typing import List, Dict, Mapping, Any
from . import datasets


@dataclass(frozen=True)
class TextureSystem:
    name: str
    polygons: Mapping[str, Any]
    meta: Mapping[str, Any]


_SYSTEMS: Dict[str, TextureSystem] = {
    "USDA": TextureSystem(
        name="USDA",
        polygons=datasets.USDA_TEXTURE_CLASSES,
        meta={
            "description": "United States Department of Agriculture (USDA) Soil Texture Classification"
        },
    ),
    "HYPRES": TextureSystem(
        name="HYPRES",
        polygons=datasets.HYPRES_TEXTURE_CLASSES,
        meta={
            "description": "The HYdraulic PRoperties of European Soils (HYPRES) is a European framework for classifying soils based on their hydrologic properties"
        },
    ),
    # additional systems can be added here
}


def get_texture_system(system_name: str) -> TextureSystem:
    """Retrieve a TextureSystem by name."""
    try:
        return _SYSTEMS[system_name]
    except KeyError:
        raise ValueError(
            f"Unknown texture system {system_name!r}. Available: {list(_SYSTEMS)}"
        )


def list_texture_systems() -> Dict[str, str]:
    """List all available texture systems."""
    return {k: v.meta.get("description", "") for k, v in _SYSTEMS.items()}
