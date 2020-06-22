#%%
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,LSTM,BatchNormalization
import numpy as np

# %%
train_dir="dataset/aerial-cactus-identification/train/train"
test_dir="dataset/aerial-cactus-identification/test/test"
train=pd.read_csv('dataset/aerial-cactus-identification/train/train.csv')
train.has_cactus=train.has_cactus.astype(str)
# %%
datagen=ImageDataGenerator(rescale=1./255)
batch_size=150
train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id', y_col='has_cactus',class_mode='binary',batch_size=batch_size, target_size=(150,150))
validation_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id', y_col='has_cactus',class_mode='binary',batch_size=50, target_size=(150,150))


# %%
model=models.Sequential()
model.add(layers.LSTM(32,activation='relu',input_shape=(2,1)))
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# %%
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
epochs=10
history=model.fit(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)

# %%
