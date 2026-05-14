import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import keras
import pandas as pd
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

labels=pd.read_csv("/home/labuser/Downloads/CV_OPENCV/Deep_Learning_CV/cifar-10/trainLabels.csv", index_col=0)

img_idx=5
print(labels.label[img_idx])
# Image.open
img_path = f"/home/labuser/Downloads/CV_OPENCV/Deep_Learning_CV/cifar-10/train/{img_idx}.png"
img = Image.open(img_path)

plt.imshow(img)
plt.axis("off")
plt.title(labels.label[img_idx])
plt.show()

from sklearn.model_selection import train_test_split
y_train,y_test=train_test_split(labels.label,test_size=0.3,random_state=42)
train_idx,test_idx=y_train.index,y_test.index

# print(y_train.index)
#reading image from training
temp=[]
for img_idx in y_train.index:
    img_path=f"/home/labuser/Downloads/CV_OPENCV/Deep_Learning_CV/cifar-10/train/{img_idx}.png"
    img=np.array(Image.open(img_path)).astype("float32")
    temp.append(img)

x_train=np.stack(temp)

temp1=[]
for img_idx in y_test.index:
    img_path=f"/home/labuser/Downloads/CV_OPENCV/Deep_Learning_CV/cifar-10/train/{img_idx}.png"
    img=np.array(Image.open(img_path)).astype("float32")
    temp1.append(img)
x_test=np.stack(temp1)

#normalize image data
x_train=x_train/255
x_test=x_test/255


encode_x=LabelEncoder()
encode_x_fit=encode_x.fit_transform(y_train)
y_train=keras.utils.to_categorical(encode_x_fit)

num_class=10

model=keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding="same",activation="relu",
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        input_shape=(32,32,3),name="conv_1"),
    keras.layers.BatchNormalization(name="BN_1"),
    keras.layers.MaxPool2D(pool_size=(2,2),name="MP_1"),

    keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding="same",activation="relu",
                        kernel_regularizer=keras.regularizers.l2(0.001),name="conv_2"),
    keras.layers.BatchNormalization(name="BN_2"),
    keras.layers.MaxPool2D(pool_size=(2,2),name="MP_2"),
    keras.layers.Flatten(name="flat"),
    keras.layers.Dense(num_class,activation="softmax",name="pred_layer")
])
print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
cpfile=r"CIFAR10_checkpoint.keras"
cb_checkpoint=keras.callbacks.ModelCheckpoint(cpfile,monitor="val_accuracy",verbose=1,save_best_only=True,mode="max")
epochs=5
model.fit(x_train,y_train,epochs=epochs,validation_split=0.2,callbacks=[cb_checkpoint])

# print(model.predict(x_test[:10]))
probs=model.predict(x_test[:10])
pred_ids=np.argmax(probs,axis=1)
pred=encode_x.inverse_transform(pred_ids)
# pred=encode_x.inverse_transform(model.predict(x_test[:10]))
act=y_test[:10]
res=pd.DataFrame([pred,act]).T
res.columns=["predicted","actual"]
print(res)


# from sklearn.metrics import accuracy_score
# train_probs=model.predict(x_train)
# train_pred_id=np.argmax(train_probs,axis=1)
# train_pred=encode_x.inverse_transform(train_pred_id)

# y_train_pred_id=np.argmax(y_train,axis=1)
# y_train_pred=encode_x.inverse_transform(y_train_pred_id)

# train_acc=accuracy_score(y_train_pred_id,train_pred)

# test_probs=model.predict(x_test)
# test_ids=np.argmax(test_probs,axis=1)
# test_pred=encode_x.inverse_transform(test_ids)

# # y_test_ids=np.argmax(y_test,axis=1)
# # y_test_pred=encode_x.inverse_transform(y_test_ids)
# test_true_labels=y_test.values

# test_acc=accuracy_score(test_true_labels,test_pred)

# print("train_accuracy",test_acc)
# print("test_accuracy",train_acc)


from sklearn.metrics import accuracy_score
import numpy as np

# Train accuracy (int vs int)
train_pred_ids = np.argmax(model.predict(x_train, verbose=0), axis=1)
train_true_ids = np.argmax(y_train, axis=1)
train_acc = accuracy_score(train_true_ids, train_pred_ids)

# Test accuracy (convert y_test strings -> ids)
test_pred_ids = np.argmax(model.predict(x_test, verbose=0), axis=1)
test_true_ids = encode_x.transform(y_test.values)   # strings -> int using same LabelEncoder
test_acc = accuracy_score(test_true_ids, test_pred_ids)

print("train_accuracy:", np.round(train_acc, 5))
print("test_accuracy :", np.round(test_acc, 5))

