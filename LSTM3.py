#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization
# %%
df = pd.read_csv("dataset/digit-recognizer/train.csv")
label_encoder = LabelEncoder().fit(df.label)
labels = label_encoder.transform(df.label)
classes = list (label_encoder.classes_)
# %%
df = df.drop(["label"], axis=1)
nb_features = len(df.columns)
nb_classes = len(classes)
#%%
scaler = StandardScaler().fit(df)
df = scaler.transform(df)
x_train, x_valid, y_train, y_valid = train_test_split(df, labels, test_size=0.3)

#%%
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
x_train = np.array(x_train).reshape(29400,784,1)
x_valid = np.array(x_valid).reshape(12600,784,1)

# %%
model= Sequential()
model.add(LSTM(512, 1, input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

# %%
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
score = model.fit(x_train,y_train,epochs=250, validation_data=(x_valid,y_valid))

# %%
