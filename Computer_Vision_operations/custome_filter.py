import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ----------------------------
# 1) Read and resize image
# ----------------------------
img = cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/balloons.jpg")
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")

resized_img = cv.resize(img, (300, 300))

# ----------------------------
# 2) Define custom filters (kernels)
# ----------------------------
custom_filter0 = np.array([[ 1,  0, -1],
                           [ 0,  0,  0],
                           [-1,  0,  1]], dtype=np.float32)

custom_filter1 = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=np.float32)

custom_filter2 = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float32)

custom_filter3 = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)

filters = [
    ("Diagonal Edge (F0)", custom_filter0),
    ("Laplacian 4-neigh (F1)", custom_filter1),
    ("Laplacian 8-neigh (F2)", custom_filter2),
    ("Sharpen (F3)", custom_filter3),
]

depths = [
    ("ddepth = -1 (uint8, clipped)", -1),
    ("ddepth = CV_32F", cv.CV_32F),
    ("ddepth = CV_64F", cv.CV_64F),
]

# ----------------------------
# 3) Helper for visualization
#    - If output is float (CV_32F/CV_64F), it can contain negatives and >255.
#    - Convert to abs + normalize to 0..255 so matplotlib can show it.
# ----------------------------
def to_displayable(img_any_depth):
    # If float output: normalize for visualization
    if img_any_depth.dtype != np.uint8:
        img_abs = np.abs(img_any_depth)
        img_norm = cv.normalize(img_abs, None, 0, 255, cv.NORM_MINMAX)
        img_u8 = img_norm.astype(np.uint8)
    else:
        img_u8 = img_any_depth

    # Convert BGR -> RGB for matplotlib
    return cv.cvtColor(img_u8, cv.COLOR_BGR2RGB)

# ----------------------------
# 4) Apply filters for each ddepth and plot results
# ----------------------------
plt.figure(figsize=(16, 18))

plot_index = 1
for f_name, kernel in filters:
    for d_name, ddepth in depths:
        # Apply convolution
        out = cv.filter2D(resized_img, ddepth, kernel)

        # Plot
        plt.subplot(4, 3, plot_index)
        plt.imshow(to_displayable(out))
        plt.title(f"{f_name}\n{d_name}", fontsize=10)
        plt.axis("off")

        plot_index += 1

plt.tight_layout()
plt.show()
