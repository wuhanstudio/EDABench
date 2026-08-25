import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from loguru import logger

from gpdl import GPDL
from congestion.model import CongestionModel


def st_machine_learning_section(designs_dir: Path, design_files):

    ml_design_option = st.selectbox(
        "Choose a design for machine learning:",
        design_files,
        key="ml_design_option",
    )

    ml_model_paths = {
        "CircuitNet Model": "models/circuitnet_10000.pth",
        "Librelane Model": "models/librelane_1000.pth",
    }
    ml_model_option = st.selectbox(
        "Choose a machine learning model:",
        list(ml_model_paths),
        key="ml_model_option",
    )
    ml_model_path = ml_model_paths[ml_model_option]

    macro_region_file = st.file_uploader("Upload the macro region", type=["png", "jpg", "jpeg"])
    rudy_heatmap_file = st.file_uploader("Upload the RUDY heatmap", type=["png", "jpg", "jpeg"])
    pin_heatmap_file = None
    if ml_model_option == "Librelane Model":
        pin_heatmap_file = st.file_uploader("Upload the pin heatmap", type=["png", "jpg", "jpeg"])

    if macro_region_file is not None:
        st.text(f"File {macro_region_file.name} uploaded successfully!")
        macro_region_path = "temp/" + macro_region_file.name
        with open(macro_region_path, "wb") as f:
            f.write(macro_region_file.getvalue())
        logger.debug(f"File saved to temp directory: {macro_region_path}")
        st.image(macro_region_path, caption=f"Macro region: {macro_region_file.name}")

    if rudy_heatmap_file is not None:
        st.text(f"File {rudy_heatmap_file.name} uploaded successfully!")
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
        and (ml_model_option == "CircuitNet Model" or pin_heatmap_file is not None)
    )

    st.button(
        "Run Machine Learning",
        type="primary",
        on_click=lambda: st.session_state.update(ml_running=True)
        if not st.session_state.ml_running
        else None,
        disabled= not required_inputs_uploaded,
    )

    if st.session_state.ml_running:
        with st.spinner("Running machine learning ...", show_time=True):
            if ml_model_option == "Librelane Model":
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = CongestionModel(device).to(device)
                model.load_state_dict(torch.load(ml_model_path, map_location=device))
                model.eval()
            else:
                model = GPDL(in_channels=2, out_channels=1)
                model.init_weights(pretrained=ml_model_path)
            model.eval()

            if ml_model_option == "Librelane Model":
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

            input_array = np.stack(input_channels, axis=0)
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).float()
            if ml_model_option == "Librelane Model":
                input_tensor = input_tensor.to(device)

            prediction = model(input_tensor)
            if ml_model_option == "Librelane Model":
                prediction = torch.sigmoid(prediction)
            prediction = prediction.float().detach().cpu().numpy()

            st.image(prediction.squeeze(), caption="Predicted heatmap", clamp=True, channels="GRAY")
            st.session_state.ml_running = False
            st.success("Machine learning completed successfully.")

    # Use Existing Classic run for comparison
    st.subheader("Use Existing Runs")
    designs_run_dir = Path("designs") / ml_design_option / "runs"
    if not os.path.exists(designs_run_dir):
        os.makedirs(designs_run_dir)

    classic_run_files = [
        f.name for f in designs_run_dir.iterdir() if f.is_dir() and f.name.startswith("Classic_")
    ]
    classic_run_option = st.selectbox("Choose a Classic run:", classic_run_files)

    from heatmap import plot_map

    gt_heatmap = None
    if classic_run_option:
        if (designs_run_dir / classic_run_option / "routing_gt.map").exists():
            gt_heatmap_path = plot_map(designs_run_dir / classic_run_option / "routing_gt.map")
            gt_heatmap = cv2.imread(gt_heatmap_path, cv2.IMREAD_GRAYSCALE)
            st.image(gt_heatmap, caption="Ground truth heatmap")
        else:
            st.error(
                f"The selected Classic run '{classic_run_option}' does not have a ground truth heatmap.",
                icon="🚨",
            )

    ml_run_files = [
        f.name for f in designs_run_dir.iterdir() if f.is_dir() and f.name.startswith("ML_")
    ]
    ml_run_option = st.selectbox("Choose a ML run:", ml_run_files)

    if ml_run_option:
        with st.spinner("Running machine learning ...", show_time=True):
            if ml_model_option == "Librelane Model":
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = CongestionModel(device).to(device)
                model.load_state_dict(torch.load(ml_model_path, map_location=device))
                model.eval()
            else:
                model = GPDL(in_channels=2, out_channels=1)
                model.init_weights(pretrained=str(ml_model_path))
                model.eval()

            macro_placement_heatmap = None
            rudy_heatmap = None
            pin_heatmap = None

            macro_region_path = designs_run_dir / ml_run_option / "placement_heatmap.png"
            rudy_heatmap_path = designs_run_dir / ml_run_option / "rudy_heatmap.png"
            pin_heatmap_path = designs_run_dir / ml_run_option / "pin_heatmap.png"
            if ml_model_option == "Librelane Model":
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

            input_array = np.stack(input_channels, axis=0)
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).float()
            prediction = model(input_tensor)
            prediction = prediction.float().detach().cpu().numpy()
            st.image(prediction.squeeze(), caption="Predicted heatmap", clamp=True, channels="GRAY")

        if classic_run_option and gt_heatmap is not None:
            prediction_scaled = cv2.normalize(prediction.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            ml_diff_heatmap = cv2.absdiff(prediction_scaled, gt_heatmap)
            st.image(ml_diff_heatmap, caption="Difference heatmap for ML run")
