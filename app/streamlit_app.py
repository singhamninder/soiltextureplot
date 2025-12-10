import streamlit as st
import pandas as pd
from soiltextureplot.triangle import SoilTextureTriangle
from soiltextureplot.systems import list_texture_systems
from matplotlib import colormaps


def main():
    st.title("Soil Texture Triangle")

    st.subheader("Upload your CSV file")

    uploaded = st.file_uploader(
        "**Upload CSV**",
        type=["csv"],
        help="CSV should have columns for sand, silt, and clay percentages.",
    )

    if uploaded is None:
        st.stop()

    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Let user specify which columns correspond to clay, sand, silt
    cols = list(df.columns)

    st.subheader("Column mapping")
    clay_col = st.selectbox("Clay column", options=[""] + cols, index=0)
    sand_col = st.selectbox("Sand column", options=[""] + cols, index=0)
    silt_col = st.selectbox("Silt column", options=[""] + cols, index=0)

    # Validate selection
    missing = []
    if not clay_col:
        missing.append("clay")
    if not sand_col:
        missing.append("sand")
    if not silt_col:
        missing.append("silt")

    if missing:
        st.warning(
            "Required columns are not configured. "
            f"Please select columns for: {', '.join(missing)}."
        )
        st.stop()  # do not proceed further

    st.subheader("Plot settings")
    systems = list_texture_systems()
    system_name = st.selectbox("Texture system", list(systems.keys()), index=0)
    st.markdown(systems[system_name])

    # Get a list of all available colormaps from matplotlib
    available_cmaps = sorted(list(colormaps))

    # Set 'Set3_r' as default if it's available
    default_cmap_index = 0
    if "Set3_r" in available_cmaps:
        default_cmap_index = available_cmaps.index("Set3_r")

    cmap_selection = st.selectbox(
        "Choose a Colormap",
        available_cmaps,
        index=default_cmap_index,
        help="Select a color map for the texture classes. Defaults to 'Set3_r'.",
    )

    size_by = st.text_input(
        "Size by (optional column name for size scaling)",
        value=None,
        help="Specify a column name to size points by that variable.",
    )

    if uploaded is not None:
        tri = SoilTextureTriangle(system_name=system_name)
        tri.load_dataframe(df, sand_col=sand_col, silt_col=silt_col, clay_col=clay_col)
        tri.classify()
        fig, ax = tri.plot(size_by=size_by or None, cmap=cmap_selection)
        st.pyplot(fig)
        st.download_button(
            "Download classified CSV",
            data=tri.df.to_csv(index=False),
            file_name="classified_soil_texture.csv",
            mime="text/csv",
        )
        st.markdown("**Preview of your classified data:**")
        st.write(tri.df.head())


if __name__ == "__main__":
    main()
