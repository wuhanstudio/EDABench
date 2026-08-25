import os
import json
from pathlib import Path

import streamlit as st

import cv2
import numpy as np
from loguru import logger

from upload import st_upload_design
from workflow import librelane_design_config, st_display_gds, st_run_librelane

import torch
from PIL import Image
from gpdl import GPDL
from congestion.model import CongestionModel

if __name__ == "__main__":
    st.set_page_config(page_title="EDABench", page_icon=":robot_face:", layout="wide")

    st.title("EDABench")

    if not os.path.exists("designs"):
        os.makedirs("designs")

    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Part 1: Upload a design zip file
    st_upload_design()

    # List files in designs directory
    designs_dir = Path("designs")
    design_files = [f.name for f in designs_dir.iterdir() if f.is_dir()]
    design_option = st.selectbox(
        "Choose a design:",
        design_files
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
    model_option = st.selectbox("Choose a machine learning model:", list(model_paths))
    model_path = model_paths[model_option]

    # Part 2: Run Librelane on the selected design
    if design_option:
        design_config = designs_dir / design_option / "config.json"
        if os.path.exists(design_config):
            try:
                config_data = librelane_design_config(design_config, workflow_option)
            except json.JSONDecodeError:
                st.error("The chosen design has an invalid config.json.", icon="🚨")
            else:
                st.text_area(
                    "config.json",
                    json.dumps(config_data, indent=2),
                    disabled=True,
                    height="content"
                )
        else:
            st.error('The chosen file is not a valid design file.', icon="🚨")

        if "running" not in st.session_state:
            st.session_state.running = False

        # Run LibreLane on the selected design
        st_run_librelane(design_option, design_config, workflow_option, model_path)

        # Display the GDS file for the selected design
        st_display_gds(design_option)

    st.divider()

    # Part 3: Run Machine Learning on the selected design
    st.header("Machine Learning")
    ml_design_option = st.selectbox(
        "Choose a design for machine learning:",
        design_files,
        key="ml_design_option",
    )

    macro_region_file = st.file_uploader("Upload the macro region", type=["png", "jpg", "jpeg"])
    rudy_heatmap_file = st.file_uploader("Upload the RUDY heatmap", type=["png", "jpg", "jpeg"])
    pin_heatmap_file = None
    if model_option == "Librelane Model":
        pin_heatmap_file = st.file_uploader("Upload the pin heatmap", type=["png", "jpg", "jpeg"])

    if macro_region_file is not None:
        st.text(f"File {macro_region_file.name} uploaded successfully!")

        #  Save the uploaded file to a temporary location
        macro_region_path = "temp/" + macro_region_file.name
        with open(macro_region_path, "wb") as f:
            f.write(macro_region_file.getvalue())

        logger.debug(f"File saved to temp directory: {macro_region_path}")
        st.image(macro_region_path, caption=f"Macro region: {macro_region_file.name}")

    if rudy_heatmap_file is not None:
        st.text(f"File {rudy_heatmap_file.name} uploaded successfully!")

        #  Save the uploaded file to a temporary location
        rudy_heatmap_path = "temp/" + rudy_heatmap_file.name
        with open(rudy_heatmap_path, "wb") as f:
            f.write(rudy_heatmap_file.getvalue())

        logger.debug(f"File saved to temp directory: {rudy_heatmap_path}")
        st.image(rudy_heatmap_path, caption=f"RUDY heatmap: {rudy_heatmap_file.name}")

    if pin_heatmap_file is not None:
        st.text(f"File {pin_heatmap_file.name} uploaded successfully!")

        pin_heatmap_path = "temp/" + pin_heatmap_file.name
        with open(pin_heatmap_path, "wb") as f:
            f.write(pin_heatmap_file.getvalue())

        logger.debug(f"File saved to temp directory: {pin_heatmap_path}")
        st.image(pin_heatmap_path, caption=f"Pin heatmap: {pin_heatmap_file.name}")

    if "ml_running" not in st.session_state:
        st.session_state.ml_running = False

    required_inputs_uploaded = (
        macro_region_file is not None
        and rudy_heatmap_file is not None
        and (model_option == "CircuitNet Model" or pin_heatmap_file is not None)
    )
    if required_inputs_uploaded:
        st.button("Run Machine Learning", type="primary", on_click=lambda: st.session_state.update(ml_running=True) if not st.session_state.ml_running else None, disabled=st.session_state.ml_running)

    if st.session_state.ml_running:
        with st.spinner("Running machine learning ...", show_time=True):
            if model_option == "Librelane Model":
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = CongestionModel(device).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            else:
                model = GPDL(in_channels=2, out_channels=1)
                model.init_weights(pretrained=model_path)
            model.eval()

            if model_option == "Librelane Model":
                input_channels = [
                    np.array(Image.open(pin_heatmap_path).convert("L"), dtype=np.float32) / 255.0,
                    np.array(Image.open(macro_region_path).convert("L"), dtype=np.float32) / 255.0,
                    np.array(Image.open(rudy_heatmap_path).convert("L"), dtype=np.float32) / 255.0,
                ]
            else:
                input_channels = [
                    cv2.imread(macro_region_path, cv2.IMREAD_GRAYSCALE),
                    cv2.imread(rudy_heatmap_path, cv2.IMREAD_GRAYSCALE),
                ]

            # Construct input tensor for GPDL model
            input = np.stack(input_channels, axis=0)

            # Convert input to torch tensor and add batch dimension
            input = torch.from_numpy(input).unsqueeze(0).float()
            if model_option == "Librelane Model":
                input = input.to(device)

            prediction = model(input)
            if model_option == "Librelane Model":
                prediction = torch.sigmoid(prediction)
            prediction = prediction.float().detach().cpu().numpy()

            st.image(prediction.squeeze(), caption="Predicted heatmap", clamp=True, channels="GRAY")

            st.session_state.ml_running = False
            st.success("Machine learning completed successfully.")

    # List existing runs for the selected design in Part 3
    designs_run_dir = Path("designs") / ml_design_option / "runs"
    if not os.path.exists(designs_run_dir):
        os.makedirs(designs_run_dir)

    classic_run_files = [f.name for f in designs_run_dir.iterdir() if f.is_dir() and f.name.startswith(("Classic_"))]
    classic_run_option = st.selectbox("Choose a Classic run:", classic_run_files)

    # If a run is selected, display the GDS file for that run
    from heatmap import plot_map
    gt_heatmap = None
    if classic_run_option:
        if (designs_run_dir / classic_run_option / "routing_gt.map").exists():
            gt_heatmap_path = plot_map(
                designs_run_dir / classic_run_option / "routing_gt.map"
            )

            gt_heatmap = cv2.imread(gt_heatmap_path, cv2.IMREAD_GRAYSCALE)
            st.image(gt_heatmap, caption=f"Ground truth heatmap")
        else:
            st.error(f"The selected Classic run '{classic_run_option}' does not have a ground truth heatmap.", icon="🚨")

    ml_run_files = [f.name for f in designs_run_dir.iterdir() if f.is_dir() and f.name.startswith(("ML_"))]
    ml_run_option = st.selectbox("Choose a ML run:", ml_run_files)

    if ml_run_option:
        import torch
        from gpdl import GPDL

        with st.spinner("Running machine learning ...", show_time=True):

            if model_option == "Librelane Model":
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = CongestionModel(device).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            else:
                model = GPDL(in_channels=2, out_channels=1)
                model.init_weights(pretrained=str(model_path))
                model.eval()

            # Read placement heatmap and RUDY heatmap from png
            macro_placement_heatmap = None
            rudy_heatmap = None
            pin_heatmap = None

            macro_region_path = designs_run_dir / ml_run_option / "placement_heatmap.png"
            rudy_heatmap_path = designs_run_dir / ml_run_option / "rudy_heatmap.png"
            pin_heatmap_path = designs_run_dir / ml_run_option / "pin_heatmap.png"
            if model_option == "Librelane Model":
                if not macro_region_path.exists():
                    st.error(f"The selected ML run '{ml_run_option}' does not have a placement heatmap.", icon="🚨")
                else:
                    macro_placement_heatmap = np.array(Image.open(macro_region_path).convert("L"), dtype=np.float32) / 255.0
                
                if not rudy_heatmap_path.exists():
                    st.error(f"The selected ML run '{ml_run_option}' does not have a RUDY heatmap.", icon="🚨")
                else:
                    rudy_heatmap = np.array(Image.open(rudy_heatmap_path).convert("L"), dtype=np.float32) / 255.0

                if not pin_heatmap_path.exists():
                    st.error(f"The selected ML run '{ml_run_option}' does not have a pin heatmap.", icon="🚨")
                else:
                    pin_heatmap = np.array(Image.open(pin_heatmap_path).convert("L"), dtype=np.float32) / 255.0

                if macro_placement_heatmap is None or rudy_heatmap is None or pin_heatmap is None:
                    st.error(f"Cannot run ML model due to missing heatmaps for the selected ML run '{ml_run_option}'.", icon="🚨")
                    st.stop()

                input_channels = [pin_heatmap, macro_placement_heatmap, rudy_heatmap]
            else:
                if not macro_region_path.exists():
                    st.error(f"The selected ML run '{ml_run_option}' does not have a placement heatmap.", icon="🚨")
                else:
                    macro_placement_heatmap = cv2.imread(macro_region_path, cv2.IMREAD_GRAYSCALE)

                if not rudy_heatmap_path.exists():
                    st.error(f"The selected ML run '{ml_run_option}' does not have a RUDY heatmap.", icon="🚨")
                else:
                    rudy_heatmap = cv2.imread(rudy_heatmap_path, cv2.IMREAD_GRAYSCALE)

                if macro_placement_heatmap is None or rudy_heatmap is None:
                    st.error(f"Cannot run ML model due to missing heatmaps for the selected ML run '{ml_run_option}'.", icon="🚨")
                    st.stop()

                input_channels = [macro_placement_heatmap, rudy_heatmap]

            # Construct input tensor for GPDL model
            input = np.stack(input_channels, axis=0)  # Shape: (2, H, W)

            # Convert input to torch tensor and add batch dimension
            input = torch.from_numpy(input).unsqueeze(0).float()  # Shape: (1, 2, H, W)

            prediction = model(input)
            prediction = prediction.float().detach().cpu().numpy()

            st.image(prediction.squeeze(), caption="Predicted heatmap", clamp=True, channels="GRAY")

        if classic_run_option:
            # Calculate the absolute difference between the predicted heatmap and the ground truth heatmap
            # Convert prediction to the same scale as gt_heatmap if necessary
            if gt_heatmap is not None:
                prediction_scaled = cv2.normalize(prediction.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                ml_diff_heatmap = cv2.absdiff(prediction_scaled, gt_heatmap)

                st.image(ml_diff_heatmap, caption=f"Difference heatmap for ML run")
