import os
import zipfile
from pathlib import Path

import subprocess
import streamlit as st

from loguru import logger

st.title("EDABench")

if not os.path.exists("temp"):
    os.makedirs("temp")

# List files in designs directory
designs_dir = Path("designs")
files = [f.name for f in designs_dir.iterdir() if f.is_dir()]
option = st.selectbox(
    "Choose a design:",
    files
)

st.write("You selected:", option)

design_config = designs_dir / option / "config.json"
if os.path.exists(design_config):
    st.text_area("config.json", open(design_config).read(), disabled=True)
else:
    st.error('The chosen file is not a valid design file.', icon="🚨")

uploaded_file = st.file_uploader("Upload a design (zip)", type=["zip"])

if uploaded_file is not None:
    st.text(f"File {uploaded_file.name} uploaded successfully!")

    #  Save the uploaded file to a temporary location
    design_file = "temp/" + uploaded_file.name
    with open(design_file, "wb") as f:
        f.write(uploaded_file.getvalue())

    logger.debug(f"File saved to temp directory: {design_file}")

    # Extract the zip file to a temporary directory
    design_folder = Path("designs/" + uploaded_file.name)
    while design_folder.suffix:
        design_folder = design_folder.with_suffix('')

    with zipfile.ZipFile(design_file, 'r') as zip_ref:
        zip_ref.extractall(design_folder)
        st.text(f"File {uploaded_file.name} extracted successfully!")

    logger.debug(f"File extracted to: {design_folder}")

if st.button("Run LibreLane"):

    cmd = [
        "python",
        "-m",
        "librelane",
        "--dockerized",
        str(design_config)
    ]

    log_box = st.empty()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Get the latest runs for the selected design
    # latest_run = max(
    #     [d for d in (designs_dir / option / "runs").iterdir() if d.is_dir()],
    #     key=lambda d: d.name
    # )

    # logger.debug(latest_run)

    # my_bar = st.progress(0, text="Running LibreLane...")

    # for d in latest_run.iterdir():
    #     if d.is_dir():
    #         st.text(f"Processing {d.name}...")

    # for chunk in iter(process.stdout.readline, ""):
    #     chunk = chunk.replace("\r", "").rstrip()

    #     # skip empty noise
    #     if not chunk:
    #         continue

    #     logger.info(chunk)

    # log_box = st.empty()
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
                "\n".join(log_lines[-10:]),  # limit memory
                language="bash"
            )

    return_code = process.wait()

    if return_code == 0:
        st.success("LibreLane completed successfully.")
    else:
        st.error(f"LibreLane failed with exit code {return_code}")
