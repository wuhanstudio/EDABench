import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load heatmap data
df = pd.read_csv("rudy.map")

# Determine original heatmap dimensions
width = df["x1"].max()
height = df["y1"].max()

# Create full-resolution heatmap
heatmap = np.zeros((256, 256), dtype=np.float32)

for _, row in df.iterrows():
    # Convert coordinates to 256x256 pixel indices
    x0 = int(round(row["x0"] / width * 256))
    x1 = int(round(row["x1"] / width * 256))
    y0 = int(round(row["y0"] / height * 256))
    y1 = int(round(row["y1"] / height * 256))

    heatmap[y0:y1, x0:x1] = row["value (%)"]

# Normalize to 0-255
heatmap -= heatmap.min()
heatmap /= heatmap.max()
heatmap = (heatmap * 255).astype(np.uint8)

plt.imshow(heatmap, cmap='plasma', interpolation='nearest')
plt.show()

# Save image
plt.imsave("heatmap_256x256.png", heatmap, cmap='plasma', format='png')
