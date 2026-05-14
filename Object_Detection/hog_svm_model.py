import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import data, color, transform

from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# -----------------------------
# 1) HOG on your own image (visualization only)
# -----------------------------
img = imread("/home/labuser/Downloads/CV_OPENCV/Object_Detection/hemant_photo.jpg")
print("Original image shape:", img.shape)

plt.figure(figsize=(6,4))
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.show()

img_resized = resize(img, (128, 64), anti_aliasing=True)
print("Resized image shape:", img_resized.shape)

fd, hog_img = hog(
    img_resized,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=-1,          # RGB channels are last axis
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True
)

print("HOG fd length:", len(fd))

plt.figure(figsize=(6,4))
plt.imshow(exposure.rescale_intensity(hog_img), cmap="gray")
plt.title("HOG visualization (hog_img)")
plt.axis("off")
plt.show()


# -----------------------------
# 2) Prepare training data: positives (faces) + negatives (random patches)
# -----------------------------
# Positive patches: LFW faces (grayscale already)
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
positive_patches = faces.images
print("Positive patches:", positive_patches.shape)  # (Npos, H, W)

patch_size = positive_patches[0].shape  # e.g., (62, 47)


# Safe grayscale conversion for skimage sample images
def to_gray(im):
    im = np.asarray(im)
    if im.ndim == 3:
        return color.rgb2gray(im)
    return im.astype(np.float32) / (255.0 if im.max() > 1.0 else 1.0)


# Background images for negatives
names = ["camera", "text", "coins", "moon", "page", "clock",
         "immunohistochemistry", "chelsea", "coffee", "hubble_deep_field"]

bg_images = []
for name in names:
    im = getattr(data, name)()
    bg_images.append(to_gray(im))

print("Background images loaded:", len(bg_images))


# Extract random negative patches from backgrounds
def extract_patches(img, N, scale=1.0, patch_size=patch_size, random_state=0):
    # scale patch size
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))

    extractor = PatchExtractor(
        patch_size=extracted_patch_size,
        max_patches=N,                 # ✅ USE N properly
        random_state=random_state
    )

    patches = extractor.transform(img[np.newaxis, ...])  # shape: (N, ph, pw)

    # resize back to patch_size so all negatives match positives
    if scale != 1.0:
        patches = np.array([
            transform.resize(p, patch_size, anti_aliasing=True)
            for p in patches
        ])
    return patches


negative_patches = np.vstack([
    extract_patches(im, N=200, scale=scale, random_state=0)
    for im in bg_images
    for scale in [0.5, 1.0, 2.0]
])

print("Negative patches:", negative_patches.shape)  # (Nneg, H, W)


# -----------------------------
# 3) Extract HOG features for ALL patches with SAME parameters
# -----------------------------
HOG_TRAIN_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True
)

def hog_feat(im2d):
    return hog(im2d, **HOG_TRAIN_PARAMS)

X = np.array([hog_feat(im) for im in np.concatenate([positive_patches, negative_patches])])
y = np.zeros(X.shape[0], dtype=int)
y[:positive_patches.shape[0]] = 1

print("X shape:", X.shape, "(rows=samples, cols=features)")
print("y shape:", y.shape)


# -----------------------------
# 4) Train SVM (with scaling + correct GridSearch)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(dual=False, max_iter=20000))
])

param_grid = {"svm__C": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]}   # ✅ correct parameter name

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best CV score:", grid.best_score_)
print("Best params:", grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print("\nValidation report:\n", classification_report(y_test, y_pred))