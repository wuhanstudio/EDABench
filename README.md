# EDABench

## OpenROAD (TCL)

LibreLane detects and imports all Python modules found that have the prefix `librelane_plugin_`.

```
nix-shell ~/librelane/shell.nix
python -m librelane designs/pm32/config.json
```

The script is available at [librelane_plugin_ml/scripts](https://github.com/wuhanstudio/EDABench/tree/main/librelane_plugin_ml/scripts)

## Streamlit GUI

```
sudo apt install libcairo-dev
```

Install Docker:

```
curl -fsSL https://get.docker.com | sh
```

Install `uv`

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a Python virtual environment:

```
uv sync
```

Run app:

```
uv run streamlit run app.py
```
