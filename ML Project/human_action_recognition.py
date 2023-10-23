import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
import random
import numpy as np
import pandas as pd
train_csv =pd.read_csv("D:\Training_set.csv")
test_csv=pd.read_csv("D:\Testing_set.csv")
train_fol=glob.glob("D:\train")
test_fol=glob.glob("D:\test")
filename = train_csv['filename']

situation = train_csv['label']
def disp():
    import cv2
    import matplotlib.pyplot as plt
    num = random.randint(1,5000)
    imgg = "Image_{}.jpg".format(num)
    train ="D:\\train\\" 
    if os.path.exists(train+imgg):
        testImage = cv2.imread(train+imgg)
        plt.imshow(testImage)
        plt.title("{}".format(train_csv.loc[train_csv['filename'] == "{}".format(imgg), 'label'].item()))

    else:
        #print(train+img)
        print("File Path not found \nSkipping the file!!")
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from PIL import Image

from tensorflow.keras.utils import to_categorical

import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array,load_img

img_data=[]
img_label=[]

for i in range(len(train_csv)):
    img="D:\\train\\"+train_csv["filename"][i]
    img=load_img(img,target_size=(160,160))
    img=img_to_array(img)
    img_data.append(img)
    img_label.append(train_csv["label"][i])

img_data=np.array(img_data)
img_label=np.array(img_label)
inp_shape = (160,160,3)
iii = img_data
iii = np.asarray(iii)
y_train = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
print(y_train[0])
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history = vgg_model.fit(iii,y_train, epochs=5)
vgg_model.save_weights("deliLab2_HARmodel.h5")
import pickle

# save the iris classification model as a pickle file
model_pkl_file = "Humanac.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(history, file)
import tensorflow as tf

# Assuming you have named your model as 'model'
model_path = 'D:\\HARmodel\\'

tf.saved_model.save(vgg_model, model_path)

from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16

# Create a new instance of the VGG16 mode
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))




# Load the saved weights into the model
vgg_model.load_weights("C:\\Users\\LAXITA\\deliLab2_HARmodel.h5")


# Function to read images as array

def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))
import matplotlib.image as img

# Function to predict

def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]

    return prediction
image = img.imread("D:\\test\\Image_2.jpg")
plt.imshow(image)
test_predict("D:\\test\\Image_2.jpg")





