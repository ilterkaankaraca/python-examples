import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, SimpleRNN, BatchNormalization
from google.colab import drive
import io
import tensorflow as tf
drive.mount('/content/drive')

df= pd.read_csv("/content/drive/My Drive/data/telefon/telefon_fiyat_degisimi.csv")
print(df)

label_encoder=LabelEncoder().fit(df.price_range)
labels=label_encoder.transform(df.price_range)
classes = list(label_encoder.classes_)

print(classes)

df = df.drop(["price_range"], axis=1)
nb_features = len(df.columns)
nb_classes = len(classes)

scaler = StandardScaler().fit(df)
df = scaler.transform(df)
x_train, x_valid, y_train, y_valid = train_test_split(df, labels, test_size=0.3)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1],1)
x_valid = np.array(x_valid).reshape(x_valid.shape[0], x_valid.shape[1],1)

model= Sequential()
model.add(Conv1D(512, 1, input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(64,1))
model.add(Activation("relu"))
model.add(SimpleRNN(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
score = model.fit(x_train,y_train,epochs=100, validation_data=(x_valid,y_valid))

print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["acc"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_acc"])))

plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"], color="g")
plt.plot(model.history.history["val_loss"], color="r")
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()