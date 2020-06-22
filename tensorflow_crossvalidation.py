import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow
from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from random import seed
seed = 7
np.random.seed(seed)
df=pd.read_csv("dataset\\telefon_fiyat_degisimi.csv")
label_encoder=LabelEncoder().fit(df.price_range)
labels=label_encoder.transform(df.price_range)
classes = list(label_encoder.classes_)

x = df.drop(["price_range"], axis=1)
y = labels
kfold= StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
sc = StandardScaler()
x = sc.fit_transform(x)
for train, test in kfold.split(x, y):
  # create model
	model = Sequential()
	model.add(Dense(16,input_dim=20, activation="relu"))
	model.add(Dense(12, activation="relu"))
	model.add(Dense(4,activation="softmax"))
	# Compile model
	y_train = to_categorical(y[train])
	y_test = to_categorical(y[test])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(x[train],y_train,validation_data=(x[test], y_test), epochs=50)
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Test"], loc="upper left")
plt.show()