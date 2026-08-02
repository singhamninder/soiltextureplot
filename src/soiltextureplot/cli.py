"""Command-line interface for soiltextureplot."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typer

from .systems import list_texture_systems
from .triangle import SoilTextureTriangle

app = typer.Typer(
    name="soiltextureplot",
    help="Classify soil samples and plot ternary texture triangles.",
    no_args_is_help=True,
)

_REQUIRED_COLS = ("sand", "silt", "clay")


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _load_triangle(
    input_path: Path,
    system: str,
    sand_col: str,
    silt_col: str,
    clay_col: str,
) -> SoilTextureTriangle:
    try:
        tri = SoilTextureTriangle(system_name=system).load_csv(
            input_path,
            sand_col=sand_col,
            silt_col=silt_col,
            clay_col=clay_col,
        )
    except (ValueError, OSError) as exc:
        _fail(str(exc))

    assert tri.df is not None
    missing = [col for col in _REQUIRED_COLS if col not in tri.df.columns]
    if missing:
        _fail(
            "Missing required columns after mapping: "
            f"{', '.join(missing)}. "
            f"Got columns: {list(tri.df.columns)}"
        )
    return tri


def _write_csv(df: pd.DataFrame, output: Optional[Path]) -> None:
    """Write DataFrame to a path, or stdout when output is None or '-'."""
    if output is None or str(output) == "-":
        typer.echo(df.to_csv(index=False), nl=False)
        return
    df.to_csv(output, index=False)


def _save_plot(
    tri: SoilTextureTriangle,
    figure_path: Path,
    *,
    size_by: Optional[str],
    cmap: str,
    color_points: str,
    show_labels: bool,
    dpi: int,
) -> None:
    matplotlib.use("Agg")
    try:
        fig, _ax = tri.plot(
            size_by=size_by,
            cmap=cmap,
            color_points=color_points,
            show_labels=show_labels,
        )
    except (ValueError, KeyError) as exc:
        _fail(str(exc))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@app.command("list-systems")
def list_systems() -> None:
    """List available soil texture classification systems."""
    systems = list_texture_systems()
    for name, description in systems.items():
        typer.echo(f"{name}: {description}")


@app.command()
def classify(
    input: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input CSV path."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path. Omit or use '-' for stdout.",
    ),
    system: str = typer.Option("USDA", "--system", help="Texture system name."),
    sand_col: str = typer.Option("sand", "--sand-col", help="Sand column name."),
    silt_col: str = typer.Option("silt", "--silt-col", help="Silt column name."),
    clay_col: str = typer.Option("clay", "--clay-col", help="Clay column name."),
) -> None:
    """Classify samples from a CSV and write a CSV with texture_class."""
    tri = _load_triangle(input, system, sand_col, silt_col, clay_col)
    try:
        classified = tri.classify()
    except ValueError as exc:
        _fail(str(exc))
    _write_csv(classified, output)


@app.command()
def plot(
    input: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input CSV path."
    ),
    output: Path = typer.Option(
        Path("texture_triangle.png"),
        "--output",
        "-o",
        help="Output figure path (format from suffix).",
    ),
    system: str = typer.Option("USDA", "--system", help="Texture system name."),
    sand_col: str = typer.Option("sand", "--sand-col", help="Sand column name."),
    silt_col: str = typer.Option("silt", "--silt-col", help="Silt column name."),
    clay_col: str = typer.Option("clay", "--clay-col", help="Clay column name."),
    size_by: Optional[str] = typer.Option(
        None, "--size-by", help="Column name for point size scaling."
    ),
    cmap: str = typer.Option("Set3_r", "--cmap", help="Colormap for texture classes."),
    color_points: str = typer.Option(
        "black", "--color-points", help="Color for sample points."
    ),
    show_labels: bool = typer.Option(
        True,
        "--show-labels/--no-show-labels",
        help="Show sample_id labels on points.",
    ),
    dpi: int = typer.Option(150, "--dpi", help="Figure DPI."),
) -> None:
    """Plot samples on a soil texture triangle and save the figure."""
    tri = _load_triangle(input, system, sand_col, silt_col, clay_col)
    _save_plot(
        tri,
        output,
        size_by=size_by,
        cmap=cmap,
        color_points=color_points,
        show_labels=show_labels,
        dpi=dpi,
    )


@app.command()
def run(
    input: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input CSV path."
    ),
    output: Path = typer.Option(
        Path("classified.csv"),
        "--output",
        "-o",
        help="Output classified CSV path.",
    ),
    figure: Path = typer.Option(
        Path("texture_triangle.png"),
        "--figure",
        "-f",
        help="Output figure path (format from suffix).",
    ),
    system: str = typer.Option("USDA", "--system", help="Texture system name."),
    sand_col: str = typer.Option("sand", "--sand-col", help="Sand column name."),
    silt_col: str = typer.Option("silt", "--silt-col", help="Silt column name."),
    clay_col: str = typer.Option("clay", "--clay-col", help="Clay column name."),
    size_by: Optional[str] = typer.Option(
        None, "--size-by", help="Column name for point size scaling."
    ),
    cmap: str = typer.Option("Set3_r", "--cmap", help="Colormap for texture classes."),
    color_points: str = typer.Option(
        "black", "--color-points", help="Color for sample points."
    ),
    show_labels: bool = typer.Option(
        True,
        "--show-labels/--no-show-labels",
        help="Show sample_id labels on points.",
    ),
    dpi: int = typer.Option(150, "--dpi", help="Figure DPI."),
) -> None:
    """Classify samples to CSV and save a texture triangle figure."""
    tri = _load_triangle(input, system, sand_col, silt_col, clay_col)
    try:
        classified = tri.classify()
    except ValueError as exc:
        _fail(str(exc))
    _write_csv(classified, output)
    _save_plot(
        tri,
        figure,
        size_by=size_by,
        cmap=cmap,
        color_points=color_points,
        show_labels=show_labels,
        dpi=dpi,
    )


if __name__ == "__main__":
    app()
