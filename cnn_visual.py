import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import random
import os

#sabitlerin belirlenmesi
image_witdh=128
image_height=128
image_size=(image_witdh,image_height)
image_channels=3

#egitim verisinin hazırlanmaıs
filenames=os.listdir("E:\Repos\python-examples\dataset\dogs-vs-cats\\train")
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
image = load_img("E:\Repos\python-examples\dataset\dogs-vs-cats\\train\\"+sample)
plt.imshow(image)