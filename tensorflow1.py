# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve

# %%
df=pd.read_csv("dataset/diyabet/diyabet.csv")
#%%
label_encoder=LabelEncoder().fit(df["IstasyonTipi"]) #0 ile 1 arasında numaralandırır
labels=label_encoder.transform(df["IstasyonTipi"]) #normalize eder
classes = list(label_encoder.classes_)#classları liste yapar

# %%
x = df.drop(["class"], axis=1)
y = labels
# %%
sc = StandardScaler() 
x = sc.fit_transform(x) # Standartize ediyor 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5) #veri setini ayırıyor
y_train = to_categorical(y_train) # 0 dan başlayıp numara haline getiriyor verisetini
y_test = to_categorical(y_test) # 0 dan başlayıp numara haline getiriyor verisetini
#%%
model = Sequential()
model.add(Dense(16,input_dim=8, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(2,activation="softmax"))
model.add(Dense(1))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train, epochs=50)
#%%
y_pred = model.predict(x_test)
y_pred =(y_pred>0.5)
y_true=y_test

#%%
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
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history = model.fit(x_train, y_train, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
# %%
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
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
