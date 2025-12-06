import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib._cm import _Set3_data  # qualitative colors
from soiltextureplot.datasets import USDA, HYPRES
import mpltern


def calculate_centroid(vertices):
    """
    Compute centroid of a 2D polygon given an (N, 2) array of vertices.
    Uses the standard shoelace formula (no np.cross).
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Close the polygon
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0

    if np.isclose(area, 0.0):
        # Fallback: simple mean if area is ~0 (degenerate polygon)
        return np.array([x.mean(), y.mean()])

    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)

    return np.array([cx, cy])


def plot_soil_texture_classes(ax, classes=None):
    """Plot soil texture classes."""

    for (key, value), color in zip(classes.items(), _Set3_data):
        tn0, tn1, tn2 = np.array(value).T
        patch = ax.fill(tn0, tn1, tn2, ec="k", fc=color, alpha=0.6, zorder=2.1)
        centroid = calculate_centroid(patch[0].get_xy())

        # last space replaced with line break
        label = key[::-1].replace(" ", "\n", 1)[::-1].capitalize()

        ax.text(
            centroid[0],
            centroid[1],
            label,
            ha="center",
            va="center",
            transform=ax.transData,
        )

    ax.taxis.set_major_locator(MultipleLocator(10.0))
    ax.laxis.set_major_locator(MultipleLocator(10.0))
    ax.raxis.set_major_locator(MultipleLocator(10.0))

    ax.taxis.set_minor_locator(AutoMinorLocator(2))
    ax.laxis.set_minor_locator(AutoMinorLocator(2))
    ax.raxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(which="both")

    # Add custom axis labels along edges

    # Sand (%) – bottom edge, centered, below triangle
    ax.text(
        -5,
        33,
        33,  # roughly mid bottom edge in (t,l,r) = (clay,sand,silt)
        "Sand (%)",
        ha="center",
        va="top",
        fontsize=12,
        rotation=0,
        transform=ax.transTernaryAxes,  # use ternary coordinate classification_system
    )

    # Clay (%) – left edge, centered, rotated up
    ax.text(
        40,
        40,
        -8,  # somewhere along left edge
        "Clay (%)",
        ha="center",
        va="center",
        fontsize=12,
        rotation=60,  # rotate along left edge
        transform=ax.transTernaryAxes,
    )

    # Silt (%) – right edge, centered, rotated down
    ax.text(
        40,
        -8,
        40,  # somewhere along right edge
        "Silt (%)",
        ha="center",
        va="center",
        fontsize=12,
        rotation=-60,  # rotate along right edge
        transform=ax.transTernaryAxes,
    )
    ax.taxis.set_ticks_position("tick2")
    ax.laxis.set_ticks_position("tick2")
    ax.raxis.set_ticks_position("tick2")


# -------------- main plotting function -------------- #
def plot_soil_samples(
    df,
    classification_system="USDA",
    sand_col="sand",
    silt_col="silt",
    clay_col="clay",
    size_by=None,
    size_min=20,
    size_max=120,
    marker="o",
    color="black",
    alpha=0.6,
):
    """
    Plot soil texture triangle with sample points.

    df: pandas DataFrame with sand, silt, clay columns (percent, summing ~100).
    """
    # Basic validation
    required = {sand_col, silt_col, clay_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    # Make sure percentages roughly sum to 100
    sums = df[sand_col] + df[silt_col] + df[clay_col]
    if not np.allclose(sums, 100.0, atol=1.0):
        print("Warning: some rows do not sum to 100±1%. Check your data.")

    # Create ternary subplot; ternary_sum=100 means axes are in percent
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="ternary", ternary_sum=100.0)

    # Plot soil texture background
    if classification_system == "USDA":
        plot_soil_texture_classes(ax, classes=USDA)
    elif classification_system == "HYPRES":
        plot_soil_texture_classes(ax, classes=HYPRES)
    else:
        raise ValueError(
            f"Unknown soil texture classification_system: {classification_system}"
        )

    # Extract sample coordinates in order (t,l,r) = (clay, sand, silt)
    t = df[clay_col].to_numpy()
    l = df[sand_col].to_numpy()
    r = df[silt_col].to_numpy()

    # size scaling
    sizes = None
    if size_by is not None and size_by in df.columns:
        vals = df[size_by].to_numpy().astype(float)

        # Handle constant or NaN-only column gracefully
        valid = np.isfinite(vals)
        if valid.any() and not np.allclose(vals[valid], vals[valid][0]):
            vmin = vals[valid].min()
            vmax = vals[valid].max()
            # Normalize to [0, 1]
            norm = (vals - vmin) / (vmax - vmin)
            # Scale to [size_min, size_max]
            sizes = size_min + norm * (size_max - size_min)
        else:
            # Fallback: constant size if no variation
            sizes = np.full_like(vals, (size_min + size_max) / 2.0)
    else:
        # Default size if no size_by provided
        sizes = size_min

    # Scatter points on top
    sc = ax.scatter(
        t,
        l,
        r,
        c=color,
        marker=marker,
        s=sizes,
        # edgecolors="white",
        # linewidths=0.5,
        alpha=alpha,
        zorder=3.0,
        # label="Samples",
    )

    plt.tight_layout()
    return fig, ax
