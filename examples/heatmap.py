import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAP_FILES = ["pin.map", "placement.map", "routing.map", "rudy.map"]
CANVAS_SIZE = 256

def infer_cell_size(df: pd.DataFrame) -> float:
    x_starts = np.sort(df["x0"].astype(float).unique())
    y_starts = np.sort(df["y0"].astype(float).unique())
    deltas = []

    if len(x_starts) > 1:
        deltas.extend(np.diff(x_starts))
    if len(y_starts) > 1:
        deltas.extend(np.diff(y_starts))

    positive_deltas = [delta for delta in deltas if delta > 0]
    if not positive_deltas:
        raise ValueError("Unable to infer cell size from map coordinates")

    return float(min(positive_deltas))


def build_heatmap(df: pd.DataFrame) -> np.ndarray:
    cell_size = infer_cell_size(df)

    width = int(np.ceil(df["x1"].astype(float).max() / cell_size))
    height = int(np.ceil(df["y1"].astype(float).max() / cell_size))

    heatmap = np.zeros((height, width), dtype=np.float32)

    for _, row in df.iterrows():
        x0 = int(round(float(row["x0"]) / cell_size))
        x1 = int(round(float(row["x1"]) / cell_size))
        y0 = int(round(float(row["y0"]) / cell_size))
        y1 = int(round(float(row["y1"]) / cell_size))

        heatmap[y0:y1, x0:x1] = float(row.iloc[-1])

    heatmap -= heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value

    return (heatmap * 255).astype(np.uint8)


def plot_map(map_path: Path) -> None:
    df = pd.read_csv(map_path)
    heatmap = build_heatmap(df)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    if heatmap.shape[0] > CANVAS_SIZE or heatmap.shape[1] > CANVAS_SIZE:
        raise ValueError(f"{map_path.name} does not fit inside a {CANVAS_SIZE}x{CANVAS_SIZE} canvas")

    y_offset = (CANVAS_SIZE - heatmap.shape[0]) // 2
    x_offset = (CANVAS_SIZE - heatmap.shape[1]) // 2
    canvas[y_offset:y_offset + heatmap.shape[0], x_offset:x_offset + heatmap.shape[1]] = heatmap

    output_path = map_path.with_name(f"{map_path.stem}_heatmap.png")

    plt.imsave(output_path, canvas, cmap="plasma", format="png")
    print(f"Saved {output_path}")

if __name__ == "__main__":
    for map_name in MAP_FILES:
        plot_map(Path(map_name))
