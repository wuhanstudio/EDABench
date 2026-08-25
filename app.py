import os
from pathlib import Path

import streamlit as st

from upload import st_upload_design

if __name__ == "__main__":
    st.set_page_config(page_title="EDABench", page_icon=":robot_face:", layout="wide")

    st.title("EDABench")

    if not os.path.exists("designs"):
        os.makedirs("designs")

    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Part 1: Upload a design zip file
    st.subheader("Part 1: Upload a Design")
    st_upload_design()

    designs_dir = Path("designs")
    design_files = [f.name for f in designs_dir.iterdir() if f.is_dir()]

    from librelane_section import st_librelane_section
    st_librelane_section(designs_dir, design_files)

    st.divider()

    from ml_section import st_machine_learning_section
    st_machine_learning_section(designs_dir, design_files)
