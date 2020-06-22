
#%%
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
#%%
train=pd.read_csv("dataset\\yaprak_veriseti\\train.csv")
test=pd.read_csv("dataset\\yaprak_veriseti\\test.csv")
#%%
#Sınıfların belirlenmesi ve etiketlenmesi
label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)
#%%
#verilerin hazırlanması özellik ve sınıf sayısının belirlenmesi
train = train.drop(["id","species"], axis=1)
test = test.drop(["id"],axis=1)
nb_features = 192
nb_classes = len(classes)
#%%
#Eğitim verisindeki verlen standartlaştırılması
scaler = StandardScaler().fit(train.values)
train=scaler.transform(train.values)
#%%
#Eğitim verisinin eğitim ve doğrulama için ayarlanaması
x_train, x_valid, y_train, y_valid = train_test_split(train,labels,test_size=0.1)
#%%
#Etiketlerin kategorilenin belirlenmesi
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
#%%
#Giriş verilerinin yeniden boyutlandırılması
x_train = np.array(x_train).reshape(891,192,1)
x_valid = np.array(x_valid).reshape(99,192,1)
#%%
#1DESA modelini oluşturulması
model = Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add((Flatten()))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()
#%%

y_pred = model.predict(x_valid)
y_pred =(y_pred>0.5)
list(y_pred)
y_true = y_train
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

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history = model.fit(x_train, y_train, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(x_train, y_train, verbose=0)
#%%
print(f1_score)
print(recall)
print(precision)

#%%
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history['loss'], color='g', label="Eğitim kaybı")
ax1.plot(model.history.history['val_loss'], color='y', label="Doğrulama kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history['accuracy'], color='g', label="Eğitim başarımı")
ax2.plot(model.history.history['val_accuracy'], color='y', label="Doğrulama başarımı")
ax2.set_xticks(np.arange(20,100,20))
plt.legend()
plt.show()