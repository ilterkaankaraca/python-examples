import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten 
from sklearn.metrics import f1_score

df=pd.read_csv("dataset\\diyabet.csv")

#Sınıfların belirlenmesi ve etiketlenmesi
label_encoder=LabelEncoder().fit(df["class"])
labels=label_encoder.transform(df["class"])
classes = list(label_encoder.classes_)
nb_classes = 2
nb_features=8
x = df.drop(["class"], axis=1)
y = labels

sc = StandardScaler()
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#Eğitim verisindeki verlen standartlaştırılması
scaler = StandardScaler().fit(x)
x=scaler.transform(x)

#Eğitim verisinin eğitim ve doğrulama için ayarlanaması
x_train, x_test, y_train, y_test = train_test_split(x,labels,test_size=0.1)

#Etiketlerin kategorilenin belirlenmesi
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Giriş verilerinin yeniden boyutlandırılması
x_train = np.array(x_train).reshape(691,8,1)
x_test = np.array(x_test).reshape(77,8,1)

#1DESA modelini oluşturulması
model = Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add((Flatten()))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

#Derleme
model.compile(loss="binary_crossentropy", optimizer="SGD",metrics=["accuracy"])

#Eğitim
model.fit(x_train,y_train,epochs=15,validation_data=(x_test, y_test))

print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarım: ", np.mean(model.history.history["accuracy"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarım: ", np.mean(model.history.history["val_accuracy"])))

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history['loss'], color='g', label="Eğitim kaybı")
ax1.plot(model.history.history['val_loss'], color='y', label="Doğrulama kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history['accuracy'], color='g', label="Eğitim başarımı")
ax2.plot(model.history.history['val_accuracy'], color='y', label="Doğrulama başarımı")
ax2.set_xticks(np.arange(20,100,20))
plt.legend()
plt.show()