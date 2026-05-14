from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#loading image
img=imread("/home/labuser/Downloads/CV_OPENCV/Object_Detection/hemant_photo.jpg")
print(img.shape)
plt.imshow(img)
plt.show()

test_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#resize
img_resized=resize(img,(128,64))
print(img_resized.shape)
plt.imshow(img_resized)
plt.show()

#HOG feature creation
fd,hog_img= hog(img_resized,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,channel_axis=-1)
print(fd)
print(len(fd))
plt.imshow(hog_img,cmap="gray")
plt.show()
# print(hog_img)

#get the inbuilt face data from skelarn
from sklearn.datasets import fetch_lfw_people
faces=fetch_lfw_people()
positive_patches=faces.images
print(positive_patches.shape)

#loading non faces images
from skimage import data, color, feature
import skimage.data
from skimage import data, transform

image_to_use=["camera","text","coins","moon","page","clock","immunohistochemistry","chelsea","coffee","hubble_deep_field"]


images = []
for name in image_to_use:
    im = getattr(data, name)()
    # ✅ convert only if RGB
    if im.ndim == 3:
        im = color.rgb2gray(im)
    images.append(im)

#finding negative patches
from sklearn.feature_extraction.image import PatchExtractor
def extract_patches(img,N,scale=1.0,patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale*np.array(patch_size)).astype(int))
    extractor=PatchExtractor(patch_size=extracted_patch_size,max_patches=N,random_state=0)
    patches=extractor.transform(img[np.newaxis])

    if scale !=1:
        patches=np.array([transform.resize(patch,patch_size) for patch in patches])
    return patches

negative_patches=np.vstack([extract_patches(im,1000,scale) for im in images for scale in [0.5,1.0,2.0]])
negative_patches.shape

#extrtact the HOG
from skimage import feature
from itertools import chain
x_train=np.array([feature.hog(im) for im in chain(positive_patches,negative_patches)])
y_train=np.zeros(x_train.shape[0])
y_train[:positive_patches.shape[0]]=1

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(LinearSVC(dual=False),{"C":[1.0,2.0,4.0,8.0]},cv=3)
grid.fit(x_train,y_train)
print(grid.best_score_)
model=grid.best_estimator_
model.fit(x_train,y_train)


def sliding_window(img,patch_size=positive_patches[0].shape,istep=2,jstep=2,scale=1.0):
    Ni, Nj=(int(scale*s) for s in patch_size)
    for i in range(0, img.shape[0]-Ni,istep):
        for j in range(0,img.shape[1]-Nj, jstep):
            patch=img[i:i+Ni, j:j+Nj]

            if scale!=1:
                patch=transform.resize(patch,patch_size)
            yield (i,j), patch

indices, patches=zip(*sliding_window(test_image))
patches_hog=np.array([feature.hog(patch) for patch in patches])
print(patches_hog.shape)
labels=model.predict(patches_hog)
print(labels.sum())

import matplotlib.pyplot as plt
import numpy as np

# ✅ Correct: subplots() returns (fig, ax)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.imshow(test_image, cmap="gray")
ax.axis("off")

# window height (Ni) and width (Nj) taken from positive patch shape
Ni, Nj = positive_patches[0].shape  # Ni = rows/height, Nj = cols/width

indices = np.array(indices)  # ensure numpy array
labels = np.array(labels)    # ensure numpy array

# indices expected shape: (num_windows, 2) where each row is [i, j]
# labels expected shape: (num_windows,) with 1 for face, 0 for non-face
for (i, j) in indices[labels == 1]:
    rect = plt.Rectangle(
        (j, i),          # (x, y) = (col, row)
        Nj, Ni,          # width, height
        edgecolor="red",
        linewidth=2,
        facecolor="none",
        alpha=0.9
    )
    ax.add_patch(rect)

plt.show()