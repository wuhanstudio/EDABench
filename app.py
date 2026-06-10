import os
import zipfile
from pathlib import Path

import subprocess
import streamlit as st

from loguru import logger

import gdstk
import cairosvg

st.title("EDABench")

if not os.path.exists("designs"):
    os.makedirs("designs")

if not os.path.exists("temp"):
    os.makedirs("temp")

# Upload a design zip file
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

# List files in designs directory
designs_dir = Path("designs")
design_files = [f.name for f in designs_dir.iterdir() if f.is_dir()]
design_option = st.selectbox(
    "Choose a design:",
    design_files
)

if design_option:
    st.write("You selected:", design_option)

    design_config = designs_dir / design_option / "config.json"
    if os.path.exists(design_config):
        st.text_area("config.json", open(design_config).read(), disabled=True)
    else:
        st.error('The chosen file is not a valid design file.', icon="🚨")
    
    
    @st.fragment
    def st_display_gds(design_option):
        # List runs for the selected design
        designs_run_dir = Path("designs") / design_option / "runs"
        run_files = [f.name for f in designs_run_dir.iterdir() if f.is_dir()]
        run_option = st.selectbox(
            "Choose a run:",
            run_files
        )
    
        design_output = designs_run_dir / run_option / "final" / "gds" / f"{design_option}.gds"
        if os.path.exists(design_output):
            with st.spinner("Generating PNG ...", show_time=True):
                # Convert GDS to SVG for display
                design_output_svg = designs_run_dir / run_option / "final" / "gds" / f"{design_option}.svg"
                if not (design_output_svg).exists():
                    library = gdstk.read_gds(design_output)
                    top_cells = library.top_level()
    
                    label_style = {(i,0):  {"fill": "none", "stroke": "red", "font-size": "0px"} for i in range(256)}
                    top_cells[0].write_svg(str(design_output_svg),
                                        label_style=label_style)
                
                # Convert SVG to PNG for display
                design_output_png = designs_run_dir / run_option / "final" / "gds" / f"{design_option}.png"
                if not (design_output_png).exists():
                    cairosvg.svg2png(url=str(design_output_svg), write_to=str(design_output_png))
    
                st.image(str(design_output_png), caption=f"{design_option} design")
        else:
            st.error('The chosen run does not have a valid output GDS file.', icon="🚨")
    
    st_display_gds(design_option)
    
    
    if "running" not in st.session_state:
        st.session_state.running = False
    
    @st.fragment
    def st_run_librelane(design_option):
        designs_run_dir = Path("designs") / design_option / "runs"
        design_config = Path("designs") / design_option / "config.json"
    
        st.button("Run LibreLane", type="primary", on_click=lambda: st.session_state.update(running=True) if not st.session_state.running else None, disabled=st.session_state.running)
    
        if st.session_state.running:
            with st.spinner("Running design ...", show_time=True):
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
                    st.session_state.running = False
                    st.success("LibreLane completed successfully.")
                    # Get the latest runs for the selected design
                    latest_run = max(
                        [d for d in (designs_run_dir).iterdir() if d.is_dir()],
                        key=lambda d: d.name
                    )
    
                    logger.debug(latest_run)
    
                    if (latest_run / "39-openroad-globalrouting").exists():
                        st.success("Global routing completed successfully.")
    
                        latest_run = max(
                            [d for d in (designs_run_dir).iterdir() if d.is_dir()],
                            key=lambda d: d.name
                        )
    
                        gdss = Path(latest_run / "final" / "gds" / f"{design_option}.gds")
                        library = gdstk.read_gds(gdss)
                        top_cells = library.top_level()
    
                        # this is to hide all layer labels
                        label_style = {(i,0):  {"fill": "none", "stroke": "red", "font-size": "0px"} for i in range(256)}
                        top_cells[0].write_svg(str(latest_run / "final" / "gds" / f"{design_option}.svg"),
                                            label_style=label_style)
                        
                        # Display the SVG file in Streamlit
                        # st.image('gcd.svg')
    
                        cairosvg.svg2png(url=str(latest_run / "final" / "gds" / f"{design_option}.svg"), write_to=str(latest_run / "final" / "gds" / f"{design_option}.png"))
                        st.image(str(latest_run / "final" / "gds" / f"{design_option}.png"), caption=f"{design_option} design")
                else:
                    st.session_state.running = False
                    st.error(f"LibreLane failed with exit code {return_code}")
    
            st.session_state.running = False
    
    st_run_librelane(design_option)
