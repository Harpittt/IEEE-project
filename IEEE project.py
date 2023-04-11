import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


data = pd.read_csv("https://github.com/Harpittt/IEEE-project/blob/cc8b72bbf736129fd75e46dac85143fefd0e2e51/Book1.csv")


features = data.iloc[:, :12]
classes = data.iloc[:, 12:]


classes = pd.get_dummies(classes)


X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(85, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
