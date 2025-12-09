import streamlit as st
import pandas as pd
from soiltextureplot.triangle import SoilTextureTriangle
from soiltextureplot.systems import list_texture_systems
from matplotlib import colormaps


def main():
    st.title("Soil Texture Triangle")

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

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV should have columns for sand, silt, and clay percentages.",
    )
    size_by = st.text_input(
        "Size by (optional column name)",
        value="bulk_density",
        help="Specify a column name to size points by that variable.",
    )

    if uploaded is not None:
        tri = SoilTextureTriangle(system_name=system_name)
        tri.load_dataframe(pd.read_csv(uploaded))
        tri.classify()
        fig, ax = tri.plot(size_by=size_by or None, cmap=cmap_selection)
        st.pyplot(fig)
        st.download_button(
            "Download classified CSV",
            data=tri.df.to_csv(index=False),
            file_name="classified_soil_texture.csv",
            mime="text/csv",
        )
        st.markdown("**Glimpse of your classified data:**")
        st.write(tri.df.head())


if __name__ == "__main__":
    main()
