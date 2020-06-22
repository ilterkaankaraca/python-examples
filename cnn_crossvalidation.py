#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten 
from random import seed
seed = 7
np.random.seed(seed)
df=pd.read_csv("dataset\\yaprak_veriseti\\train.csv")

#%%
#Sınıfların belirlenmesi ve etiketlenmesi
label_encoder = LabelEncoder().fit(df.species)
labels = label_encoder.transform(df.species)
classes = list(label_encoder.classes_)
kfold= StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
#verilerin hazırlanması özellik ve sınıf sayısının belirlenmesi
x = df.drop(["id","species"], axis=1)
y = labels
nb_features = 192
nb_classes = len(classes)

#Eğitim verisindeki verlen standartlaştırılması
scaler = StandardScaler().fit(x.values)
x=scaler.transform(x.values)

for train, test in kfold.split(x, y):
  # create model
    model = Sequential()
    model.add(Conv1D(512,1,input_shape=(nb_features,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add((Flatten()))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(nb_classes, activation="softmax"))
    # Compile model
    y_train = to_categorical(y[train])
    y_test = to_categorical(y[test])
    x_train = np.array(x[train]).reshape(891,192,1)
    x_test = np.array(x[test]).reshape(99,192,1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
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

# %%
