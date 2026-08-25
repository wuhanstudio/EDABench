import os
from pathlib import Path

import streamlit as st

if __name__ == "__main__":
    st.set_page_config(page_title="EDABench", page_icon=":robot_face:")

    st.title("EDABench")

    if not os.path.exists("designs"):
        os.makedirs("designs")

    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Part 1: Upload a design zip file
    st.subheader("Part 1: Upload a Design", divider=True)

    from upload import st_upload_design
    st_upload_design()

    designs_dir = Path("designs")
    design_files = [f.name for f in designs_dir.iterdir() if f.is_dir()]

    # Part 2: LibreLane Flow
    st.subheader("Part 2: LibreLane Flow", divider=True)

    from librelane_section import st_librelane_section
    st_librelane_section(designs_dir, design_files)

    # Part 3: Machine Learning
    st.subheader("Part 3: Machine Learning", divider=True)

    from ml_section import st_machine_learning_section
    st_machine_learning_section(designs_dir, design_files)
