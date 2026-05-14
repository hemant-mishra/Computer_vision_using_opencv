import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.transform import resize, pyramid_gaussian
from skimage.color import rgb2gray

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ----------------------------
# CONFIG
# ----------------------------
POS_DIR = "dataset/faces"
NEG_DIR = "dataset/nonfaces"

# For face detection, common patch sizes: (64,64) or (80,80).
# Keep it fixed because HOG vector length must be constant.
WINDOW_SIZE = (64, 64)   # (H, W)
STEP_SIZE = 8            # sliding window stride in pixels

# Pyramid scaling (smaller -> faster, larger -> more precise)
PYR_SCALE = 1.25
MIN_SIZE = (64, 64)      # smallest pyramid image size

# HOG parameters (standard-ish)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True
)

# Negative patch extraction (if NEG_DIR contains background images)
NEG_PATCHES_PER_BG = 20


# ----------------------------
# UTIL: load image paths
# ----------------------------
def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


# ----------------------------
# UTIL: read -> grayscale float [0..1] -> resize fixed -> HOG
# ----------------------------
def read_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rgb2gray(img)  # float [0..1]
    return img


def hog_feature_from_patch(patch_gray, window_size=WINDOW_SIZE):
    # ensure fixed size
    patch_resized = resize(patch_gray, window_size, anti_aliasing=True)
    feat = hog(patch_resized, **HOG_PARAMS)
    return feat


# ----------------------------
# Build NEGATIVE patches from background images
# ----------------------------
def extract_random_patches(img_gray, patch_size, n_patches=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    H, W = img_gray.shape
    ph, pw = patch_size

    patches = []
    if H < ph or W < pw:
        return patches

    for _ in range(n_patches):
        y = rng.integers(0, H - ph + 1)
        x = rng.integers(0, W - pw + 1)
        patch = img_gray[y:y+ph, x:x+pw]
        patches.append(patch)
    return patches


# ----------------------------
# Build dataset X, y
# ----------------------------
def build_dataset(pos_dir, neg_dir):
    pos_paths = list_images(pos_dir)
    neg_paths = list_images(neg_dir)

    X = []
    y = []

    # POS: assume these are face patches (any size) -> resize -> hog
    print(f"[INFO] Positive images: {len(pos_paths)}")
    for p in pos_paths:
        img = read_gray(p)
        feat = hog_feature_from_patch(img)   # treated as patch
        X.append(feat)
        y.append(1)

    # NEG: could be patches or background images
    print(f"[INFO] Negative images: {len(neg_paths)}")
    rng = np.random.default_rng(123)

    for p in neg_paths:
        img = read_gray(p)

        # If already patch-like: compute hog directly
        # If big background: extract random patches
        if img.shape[0] <= WINDOW_SIZE[0] * 1.2 and img.shape[1] <= WINDOW_SIZE[1] * 1.2:
            feat = hog_feature_from_patch(img)
            X.append(feat)
            y.append(0)
        else:
            patches = extract_random_patches(img, WINDOW_SIZE, n_patches=NEG_PATCHES_PER_BG, rng=rng)
            for patch in patches:
                feat = hog_feature_from_patch(patch)
                X.append(feat)
                y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[INFO] X shape: {X.shape}  (rows=samples, cols=HOG dims)")
    print(f"[INFO] y shape: {y.shape}")
    return X, y


# ----------------------------
# Train Linear SVM
# ----------------------------
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=0.01, class_weight="balanced", max_iter=10000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[INFO] Validation report:")
    print(classification_report(y_test, y_pred))

    return model


# ----------------------------
# NMS: Non-Maximum Suppression
# boxes: list of [x1,y1,x2,y2], scores: list
# ----------------------------
def nms(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # sort by score desc

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


# ----------------------------
# Sliding Window Generator
# ----------------------------
def sliding_window(img, step_size, window_size):
    H, W = img.shape[:2]
    wh, ww = window_size
    for y in range(0, H - wh + 1, step_size):
        for x in range(0, W - ww + 1, step_size):
            yield (x, y, img[y:y+wh, x:x+ww])


# ----------------------------
# Detect faces in a given image
# ----------------------------
def detect_faces(image_path, model, score_thresh=0.5):
    # load original image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(rgb)  # float [0..1]

    boxes = []
    scores = []

    # pyramid
    for scaled in pyramid_gaussian(gray, downscale=PYR_SCALE):
        if scaled.shape[0] < MIN_SIZE[0] or scaled.shape[1] < MIN_SIZE[1]:
            break

        scale_y = gray.shape[0] / scaled.shape[0]
        scale_x = gray.shape[1] / scaled.shape[1]

        # slide window
        for (x, y, patch) in sliding_window(scaled, STEP_SIZE, WINDOW_SIZE):
            feat = hog_feature_from_patch(patch)

            # decision_function gives distance to hyperplane
            score = model.decision_function([feat])[0]

            if score > score_thresh:
                # map box back to original image coordinates
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + WINDOW_SIZE[1]) * scale_x)
                y2 = int((y + WINDOW_SIZE[0]) * scale_y)

                boxes.append([x1, y1, x2, y2])
                scores.append(score)

    # apply NMS
    keep = nms(boxes, scores, iou_thresh=0.3)
    final_boxes = [boxes[i] for i in keep]
    final_scores = [scores[i] for i in keep]

    # draw results
    out = bgr.copy()
    for (box, sc) in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{sc:.2f}", (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    count = len(final_boxes)
    print(f"[RESULT] Faces detected: {count}")

    # show
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(f"Detections: {count}")
    plt.axis("off")
    plt.show()

    return final_boxes, final_scores


# ----------------------------
# MAIN RUN
# ----------------------------
if __name__ == "__main__":
    # 1) Build training data
    X, y = build_dataset(POS_DIR, NEG_DIR)

    # 2) Train SVM
    model = train_svm(X, y)

    # 3) Detect in a test image
    test_image = "image.png"   # change to your test file
    detect_faces(test_image, model, score_thresh=0.8)