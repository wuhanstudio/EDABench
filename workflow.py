import os
import json
from pathlib import Path

import streamlit as st
from loguru import logger

import gdstk
import cairosvg
import subprocess

import cv2
import torch
import numpy as np
from gpdl import GPDL

ML_FLOW_META = {
    "flow": "Classic",
    "substituting_steps": {
        "+OpenROAD.GlobalRouting": "OpenROAD.ExportCongestionMap",
        "+Misc.ReportManufacturability": "OpenROAD.ExportGroundTruth"
    },
}

def librelane_design_config(design_config: Path, workflow_option: str):
    """
    Generate a new config.json file based on the selected workflow option.

    Args:
        design_config (Path): The path to the original config.json file.
        workflow_option (str): The selected workflow option ("Classic" or "ML Congestion Map").
    """
    with open(design_config, encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    if workflow_option == "Classic":
        config_data.pop("meta", None)
        config_data["flow"] = "Classic"
    else:
        config_data.pop("flow", None)
        config_data["meta"] = ML_FLOW_META

    with open(design_config, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, indent=2)
        config_file.write("\n")

    return config_data


@st.fragment
def st_display_gds(design_option):
    """
    Display the GDS file for the selected design.

    Args:
        design_option (str): The name of the selected design.
    """

    # List existing runs for the selected design
    designs_run_dir = Path("designs") / design_option / "runs"
    if not os.path.exists(designs_run_dir):
        os.makedirs(designs_run_dir)

    run_files = [f.name for f in designs_run_dir.iterdir() if f.is_dir()]
    run_option = st.selectbox("Choose a run:", run_files)

    # If a run is selected, display the GDS file for that run
    if run_option:
        design_output = (
            designs_run_dir / run_option / "final" / "gds" / f"{design_option}.gds"
        )
        if os.path.exists(design_output):
            with st.spinner("Generating PNG ...", show_time=True):
                # Convert GDS to SVG for display
                design_output_svg = (
                    designs_run_dir
                    / run_option
                    / "final"
                    / "gds"
                    / f"{design_option}.svg"
                )

                # Check if the SVG file already exists to avoid redundant conversion
                if not (design_output_svg).exists():
                    library = gdstk.read_gds(design_output)
                    top_cells = library.top_level()

                    label_style = {
                        (i, 0): {"fill": "none", "stroke": "red", "font-size": "0px"}
                        for i in range(256)
                    }
                    top_cells[0].write_svg(
                        str(design_output_svg), label_style=label_style
                    )

                # Convert SVG to PNG for display
                design_output_png = (
                    designs_run_dir
                    / run_option
                    / "final"
                    / "gds"
                    / f"{design_option}.png"
                )

                # Check if the PNG file already exists to avoid redundant conversion
                if not (design_output_png).exists():
                    cairosvg.svg2png(
                        url=str(design_output_svg), write_to=str(design_output_png)
                    )

                st.image(str(design_output_png), caption=f"{design_option} design")
        else:
            st.error("The chosen run does not have a valid output GDS file.", icon="🚨")


@st.fragment
def st_run_librelane(design_option, design_config, workflow_option):
    """
    Run LibreLane for the selected design.

    Args:
        design_option (str): The name of the selected design.
        design_config (str): The content of the design config file (JSON).
        workflow_option (str): The selected workflow option ("Classic" or "ML Congestion Map").
    """

    designs_run_dir = Path("designs") / design_option / "runs"
    design_config_data = librelane_design_config(design_config, workflow_option)

    # Save the updated config.json to the run directory
    design_config_json = Path("designs") / design_option / "config.json"
    with open(design_config_json, "w") as f:
        json.dump(design_config_data, f)

    st.button(
        "Run LibreLane",
        type="primary",
        on_click=lambda: (
            st.session_state.update(running=True)
            if not st.session_state.running
            else None
        ),
        disabled=st.session_state.running,
    )

    if st.session_state.running:
        with st.spinner("Running design ...", show_time=True):
            # Remove existing .map files in the current directory
            for map_file in ["placement.map", "rudy.map", "routing_gt.map"]:
                if Path(map_file).exists():
                    Path(map_file).unlink()

            cmd = ["python", "-m", "librelane", "--dockerized", str(design_config_json)]
            designs_run_dir.mkdir(parents=True, exist_ok=True)

            log_box = st.empty()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # cwd=designs_run_dir,
            )

            log_lines = []

            import re

            ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

            def strip_ansi(text: str) -> str:
                return ANSI_ESCAPE.sub("", text)

            for line in iter(process.stdout.readline, ""):
                clean = strip_ansi(line.replace("\r", "")).rstrip()

                if clean:
                    log_lines.append(clean)

                    log_box.code(
                        "\n".join(log_lines[-10:]), language="bash"  # limit memory
                    )

            return_code = process.wait()

            if return_code == 0:
                st.session_state.running = False
                st.success("LibreLane completed successfully.")
                # Get the latest runs for the selected design
                latest_run = max(
                    [d for d in (designs_run_dir).iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                )

                logger.debug(latest_run)
                if Path("placement.map").exists() and Path("rudy.map").exists():
                    logger.debug("Found placement.map and rudy.map in the current directory.")
                    st.success("Found placement.map and rudy.map.")

                    # Move the heatmap files to the latest run directory
                    (latest_run / "placement.map").write_bytes(Path("placement.map").read_bytes())
                    (latest_run / "rudy.map").write_bytes(Path("rudy.map").read_bytes())

                    from heatmap import plot_map
                    placement_heatmap_path = plot_map(
                        latest_run / "placement.map"
                    )
                    rudy_heatmap_path = plot_map(
                        latest_run / "rudy.map"
                    )

                    # Read placement heatmap and RUDY heatmap from png
                    macro_placement_heatmap = cv2.imread(placement_heatmap_path, cv2.IMREAD_GRAYSCALE)
                    rudy_heatmap = cv2.imread(rudy_heatmap_path, cv2.IMREAD_GRAYSCALE)

                    col1, col2, col3 = st.columns(3)
 
                    with col1:
                        st.image(macro_placement_heatmap, caption=f"Macro placement heatmap")

                    with col2:
                        st.image(rudy_heatmap, caption=f"RUDY heatmap")

                    # Construct input tensor for GPDL model
                    input = np.stack([macro_placement_heatmap, rudy_heatmap], axis=0)  # Shape: (2, H, W)

                    model = GPDL(in_channels=2, out_channels=1)
                    model.init_weights(pretrained="models/circuitnet_10000.pth")
                    model.eval()

                    # Convert input to torch tensor and add batch dimension
                    input = torch.from_numpy(input).unsqueeze(0).float()  # Shape: (1, 2, H, W)

                    prediction = model(input)
                    prediction = prediction.float().detach().cpu().numpy()

                    # Plot the prediction heatmap
                    with col3:
                        st.image(prediction.squeeze(), caption="Predicted congestion heatmap", clamp=True, channels="GRAY")

                if Path("routing_gt.map").exists():
                    logger.debug("Found routing_gt.map in the current directory.")
                    st.success("Found routing_gt.map.")

                    # Move the ground truth file to the latest run directory
                    (latest_run / "routing_gt.map").write_bytes(Path("routing_gt.map").read_bytes())

                    gt_heatmap_path = plot_map(
                        latest_run / "routing_gt.map"
                    )

                    gt_heatmap = cv2.imread(gt_heatmap_path, cv2.IMREAD_GRAYSCALE)
                    st.image(gt_heatmap, caption=f"Ground truth heatmap")

                    # Calculate the absolute difference between the predicted heatmap and the ground truth heatmap
                    # Convert prediction to the same scale as gt_heatmap if necessary
                    prediction_scaled = cv2.normalize(prediction.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                    diff_heatmap = cv2.absdiff(prediction_scaled, gt_heatmap)

                    st.image(diff_heatmap, caption=f"Difference heatmap")

                # Display the GDS file for the latest run
                if (latest_run / "39-openroad-globalrouting").exists():
                    st.success("Global routing completed successfully.")

                    latest_run = max(
                        [d for d in (designs_run_dir).iterdir() if d.is_dir()],
                        key=lambda d: d.name,
                    )

                    gdss = Path(latest_run / "final" / "gds" / f"{design_option}.gds")
                    library = gdstk.read_gds(gdss)
                    top_cells = library.top_level()

                    # this is to hide all layer labels
                    label_style = {
                        (i, 0): {"fill": "none", "stroke": "red", "font-size": "0px"}
                        for i in range(256)
                    }
                    top_cells[0].write_svg(
                        str(latest_run / "final" / "gds" / f"{design_option}.svg"),
                        label_style=label_style,
                    )

                    # Display the SVG file in Streamlit
                    # st.image('gcd.svg')

                    cairosvg.svg2png(
                        url=str(latest_run / "final" / "gds" / f"{design_option}.svg"),
                        write_to=str(
                            latest_run / "final" / "gds" / f"{design_option}.png"
                        ),
                    )

                    # st.image(
                    #     str(latest_run / "final" / "gds" / f"{design_option}.png"),
                    #     caption=f"{design_option} design",
                    # )
            else:
                st.session_state.running = False
                st.error(f"LibreLane failed with exit code {return_code}")

        st.session_state.running = False
