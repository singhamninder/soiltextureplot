import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Soil texture plotting (Marimo pilot)

        This notebook mirrors `texture_plot.ipynb` while the project keeps Jupyter +
        `nbqa` support during the pilot.
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    from soiltexture import getTexture
    from soiltextureplot.systems import list_texture_systems
    from soiltextureplot.triangle import SoilTextureTriangle

    return SoilTextureTriangle, getTexture, list_texture_systems, pd


@app.cell
def _(pd):
    # Example dataset matching texture_plot.ipynb
    example_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            "sand": [65, 70, 75, 80, 35, 99.8, 97.2],
            "silt": [20, 24, 21, 16, 45, 0.2, 2.8],
            "clay": [15, 6, 4, 4, 20, 0.0, 0.0],
            "BD": [1.002, 1.277, 1.008, 1.927, 1.774, 1.66, 1.68],
        }
    )
    example_df
    return (example_df,)


@app.cell
def _(example_df, getTexture):
    texture_class = example_df.apply(
        lambda row: getTexture(row["sand"], row["clay"]), axis=1
    )
    texture_class
    return (texture_class,)


@app.cell
def _(SoilTextureTriangle, example_df):
    tri = SoilTextureTriangle(system_name="USDA")
    tri.load_dataframe(example_df)
    fig, ax = tri.plot(size_by="BD", cmap="viridis")
    fig
    return ax, fig, tri


@app.cell
def _(tri):
    tri.classify()
    return


@app.cell
def _(list_texture_systems):
    list_texture_systems()
    return


if __name__ == "__main__":
    app.run()
