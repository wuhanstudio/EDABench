import zipfile
from pathlib import Path

import streamlit as st
from loguru import logger


@st.fragment
def st_upload_design():
    uploaded_file = st.file_uploader("Upload a design (zip)", type=["zip"])

    if uploaded_file is not None:
        st.text(f"File {uploaded_file.name} uploaded successfully!")

        #  Save the uploaded zip file to a temporary location
        design_file = "temp/" + uploaded_file.name
        with open(design_file, "wb") as f:
            f.write(uploaded_file.getvalue())

        logger.debug(f"File saved to temp directory: {design_file}")

        # Extract the zip file to a temporary directory
        design_folder = Path("designs/" + uploaded_file.name)
        while design_folder.suffix:
            design_folder = design_folder.with_suffix("")

        with zipfile.ZipFile(design_file, "r") as zip_ref:
            zip_ref.extractall(design_folder)
            st.text(f"File {uploaded_file.name} extracted successfully!")

        logger.debug(f"File extracted to: {design_folder}")
