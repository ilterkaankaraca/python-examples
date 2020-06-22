#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,LSTM,BatchNormalization
# %%
df=pd.read_csv("dataset/telefon/telefon_fiyat_degisimi.csv")
label_encoder=LabelEncoder().fit(df.price_range)
labels=label_encoder.transform(df.price_range)
classes = list(label_encoder.classes_)

#%%
x = df.drop(["price_range"], axis=1)
y = labels
nb_features = len(x.columns)
nb_classes = len(classes)
sc = StandardScaler() 
x = sc.fit_transform(x) # Standartize ediyor 
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2) #veri setini ayırıyor
y_train = to_categorical(y_train) # 0 dan başlayıp numara haline getiriyor verisetini
y_valid = to_categorical(y_valid) # 0 dan başlayıp numara haline getiriyor verisetini

#%%
x_train  = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1],1)
x_valid  = np.array(x_valid).reshape(x_valid.shape[0], x_valid.shape[1],1)

#%%
model = Sequential()
model.add(LSTM(512,input_shape = (nb_features,1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(2048, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
score = model.fit(x_train, y_train, epochs = 100, validation_data=(x_valid,y_valid))

#%%
print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["acc"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_acc"])))
print(("Ortalama F1-Skor Değeri: ", np.mean(model.history.history["val_f1_m"])))
print(("Ortalama Kesinlik Değeri: ", np.mean(model.history.history["val_precision_m"])))
print(("Ortalama Duyarlılık Değeri: ", np.mean(model.history.history["val_recall_m"])))

# %%
import matplotlib.pyplot as plt
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

plt.plot(model.history.history["f1_m"], color="y")
plt.plot(model.history.history["val_f1_m"], color="b")
plt.title("Model F1-Skor")
plt.ylabel("F1-Skor")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()
