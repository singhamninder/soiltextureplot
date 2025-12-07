import streamlit as st
import pandas as pd
from soiltextureplot.triangle import SoilTextureTriangle
from soiltextureplot.systems import list_texture_systems


def main():
    st.title("Soil Texture Triangle")

    systems = list_texture_systems()
    system_name = st.selectbox("Texture system", list(systems.keys()), index=0)

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV should have columns for sand, silt, and clay percentages.",
    )
    size_by = st.text_input("Size by (optional column name)", value="bulk_density")

    if uploaded is not None:
        tri = SoilTextureTriangle(system_name=system_name)
        tri.load_dataframe(pd.read_csv(uploaded))
        tri.classify()
        fig, ax = tri.plot(size_by=size_by or None)
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
