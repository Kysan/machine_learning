
import random
from keras import Sequential
from keras.layers import Dense


def genDataSet():
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    numbers = list(range(0, 100))
    random.shuffle(numbers)

    for i in numbers:
        x_train.append(i)
        y_test.append(i*2+1)

    numbers_test = list(range(100, 150))
    random.shuffle(numbers_test)
    for i in numbers_test:
        x_train.append(i)
        y_test.append(i*2+1)

    return (x_train, y_train), (x_test, y_test)


(xs, ys), (xt, yt) = genDataSet()

model = Sequential()


# un Dense layer est un layer
# ou chaque noeud est connecté à chacun des noeuds de la couche suivante


# units <= nombre de noeud du layer
model.add(Dense(units=1, activation='sigmoid',
          input_shape=(1, 1), name="input"))

# invisble layer
model.add(Dense(units=3, activation='sigmoid'))

# out layer
model.add(Dense(units=1, activation='rle', name="output"))


# model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd'
)


model.fit()
