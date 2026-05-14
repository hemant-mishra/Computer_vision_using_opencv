import cv2 as cv
import matplotlib.pyplot as plt

# 1) Read image
img = cv.imread("/home/labuser/Downloads/CV_OPENCV/Object_Detection/hemant_photo.jpg")
if img is None:
    raise FileNotFoundError("Image not found. Check the path.")

# 2) Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3) Load cascades
face_cascade = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_frontalface_default.xml")
eye_cascade  = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_eye.xml")
smile_cascade = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_smile.xml")

# Safety check
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    raise IOError("One or more cascade XML files not loaded. Check file paths.")

# 4) Detect faces (use gray)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 5) Draw results on a copy
out = img.copy()

for (x, y, w, h) in faces:
    # Draw face rectangle
    cv.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # ROI for face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = out[y:y+h, x:x+w]

    # 6) Detect eyes within face ROI
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 7) Detect smile within face ROI (smile needs different params usually)
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)

# 8) Show output (convert BGR->RGB for matplotlib)
plt.figure(figsize=(8, 8))
plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()