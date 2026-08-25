import json
from pathlib import Path

import streamlit as st

from workflow import librelane_design_config, st_display_gds, st_run_librelane


def st_librelane_section(designs_dir: Path, design_files):

    design_option = st.selectbox(
        "Choose a design:",
        design_files,
        key="librelane_design_option",
    )

    workflow_option = st.radio(
        "Choose the librelane workflow:",
        ["Classic", "ML Congestion Map"],
        captions=[
            "Default librelane workflow.",
            "Use machine learning to generate a congestion map for librelane.",
        ],
    )

    model_paths = {
        "CircuitNet Model": "models/circuitnet_10000.pth",
        "Librelane Model": "models/librelane_1000.pth",
    }
    model_path = None
    if workflow_option == "ML Congestion Map":
        model_option = st.selectbox(
            "Choose a machine learning model:",
            list(model_paths),
            key="librelane_model_option",
        )
        model_path = model_paths[model_option]

    if design_option:
        design_config = designs_dir / design_option / "config.json"
        if design_config.exists():
            try:
                config_data = librelane_design_config(design_config, workflow_option)
            except json.JSONDecodeError:
                st.error("The chosen design has an invalid config.json.", icon="🚨")
            else:
                st.text_area(
                    "config.json",
                    json.dumps(config_data, indent=2),
                    disabled=True,
                    height="content",
                )
        else:
            st.error("The chosen file is not a valid design file.", icon="🚨")

        if "running" not in st.session_state:
            st.session_state.running = False

        st_run_librelane(design_option, design_config, workflow_option, model_path)
        st_display_gds(design_option)
