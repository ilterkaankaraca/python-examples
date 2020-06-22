#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import matplotlib.pyplot as plt
# %%
le = LabelEncoder()
df=pd.read_csv("dataset/telefon/telefon_fiyat_degisimi.csv")
#%%
label_encoder = le.fit(df["class"])
labels = label_encoder.transform(df["class"])
classes = list(label_encoder.classes_)
#%%
x = df.drop(["class"], axis=1)
y = labels
nb_features = x.shape[1]
nb_classes = len(classes)
# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3)
#%%
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#%%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, SimpleRNN, BatchNormalization

model = Sequential()
model.add(SimpleRNN(512, input_shape=(nb_features, 1)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

# %%
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])
#%%
score = model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test))

#%%
print("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["acc"]))
print("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_acc"]))

# %%
import matplotlib.pyplot as plt
plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Test"], loc="upper left")
plt.show()

# %%
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Test"], loc="upper left")
plt.show()
