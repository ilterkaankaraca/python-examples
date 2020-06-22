import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Activation,BatchNormalization,Dense
#sabitlerin belirlenmesi
image_width=128
image_height=128
image_size=(image_width,image_height)
image_channels=3

#egitim verisinin hazırlanmaıs
filenames=os.listdir("E:/Repos/python-examples/dataset/dogs-vs-cats/train")
categories = []
for i in filenames:
    category = i.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)
df = pd.DataFrame({"filename": filenames, "category": categories})

#Veriyi görselleştirmek
df["category"].value_counts().plot.bar()


#ratgele bir görüntü seçilmesi 
sample = random.choice(filenames)
image = load_img("E:/Repos/python-examples/dataset/dogs-vs-cats/train/"+sample)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(image_width,image_height,image_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))
model.summary()
                 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
df["category"]=df["category"].replace({0: "cat", 1: "dog"})
train_df, validate_df = train_test_split(df,test_size=0.2)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15
train_datagen=ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "E:\Repos\python-examples\dataset\dogs-vs-cats\\train",
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size)
validation_datagen=ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "E:\Repos\python-examples\dataset\dogs-vs-cats\\train",
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size)
epochs=10
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size)
model.save_weights("model1.h5")
predict=model.predict_generaotr(test_generator, steps=np.ceil(nb_samples/batch_size))
