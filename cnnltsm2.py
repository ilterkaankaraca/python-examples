#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization
from google.colab import drive
import io
import tensorflow as tf
drive.mount('/content/drive')
#%%
df= pd.read_csv("/content/drive/My Drive/data/credit/creditcard.csv")
print(df)
#%%
label_encoder=LabelEncoder().fit(df.Class)
labels=label_encoder.transform(df.Class)
classes = list(label_encoder.classes_)
#%%
print(classes)

df = df.drop(["Class"], axis=1)
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
model.add(Conv1D(128, 1, input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(64,1))
model.add(Activation("relu"))
model.add(LSTM(32))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
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

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc',f1_m,precision_m, recall_m, tf.keras.metrics.AUC()])
score = model.fit(x_train,y_train,epochs=10, validation_data=(x_valid,y_valid))

print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["acc"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_acc"])))
print(("Ortalama Eğitim F1-Skor Değeri: ", np.mean(model.history.history["f1_m"])))
print(("Ortalama Doğrulama F1-Skor Değeri: ", np.mean(model.history.history["val_f1_m"])))
print(("Ortalama Eğitim Kesinlik Değeri: ", np.mean(model.history.history["precision_m"])))
print(("Ortalama Doğrulama Kesinlik Değeri: ", np.mean(model.history.history["val_precision_m"])))
print(("Ortalama Eğitim Duyarlılık Değeri: ", np.mean(model.history.history["recall_m"])))
print(("Ortalama Doğrulama Duyarlılık Değeri: ", np.mean(model.history.history["val_recall_m"])))
print(("Ortalama Eğitim AUC Değeri: ", np.mean(model.history.history["auc"])))
print(("Ortalama Doğrulama AUC Değeri: ", np.mean(model.history.history["val_auc"])))

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

plt.plot(model.history.history["precision_m"], color="y")
plt.plot(model.history.history["val_precision_m"], color="b")
plt.title("Model Kesinlik")
plt.ylabel("Kesinlik")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()

plt.plot(model.history.history["recall_m"], color="y")
plt.plot(model.history.history["val_recall_m"], color="b")
plt.title("Model Duyarlılık")
plt.ylabel("Duyarlılık")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()

plt.plot(model.history.history["auc"], color="y")
plt.plot(model.history.history["val_auc"], color="b")
plt.title("Model AUC")
plt.ylabel("AUC")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc="upper left")
plt.show()

# %%
