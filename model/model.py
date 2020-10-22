import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras import optimizers

number = 100

# お使いの仮想環境のディレクトリ構造等によってファイルパスは異なります。
path_aji = os.listdir('../../fish_img/aji')
path_chinu = os.listdir('../../fish_img/chinu')
path_kasago = os.listdir('../../fish_img/kasago')
path_kijihata = os.listdir('../../fish_img/kijihata')
path_seabass = os.listdir('../../fish_img/seabass')

img_aji = []
img_chinu = []
img_kasago = []
img_kijihata = []
img_seabass = []

for i in range(len(path_aji)):
    img = cv2.imread('../../fish_img/aji/' + path_aji[i])
    img = cv2.resize(img, (50,50))
    img_aji.append(img)

for i in range(len(path_chinu)):
    img = cv2.imread('../../fish_img/chinu/' + path_chinu[i])
    img = cv2.resize(img, (50,50))
    img_chinu.append(img)

for i in range(len(path_kasago)):
    img = cv2.imread('../../fish_img/kasago/' + path_kasago[i])
    img = cv2.resize(img, (50,50))
    img_kasago.append(img)

for i in range(len(path_kijihata)):
    img = cv2.imread('../../fish_img/kijihata/' + path_kijihata[i])
    img = cv2.resize(img, (50,50))
    img_kijihata.append(img)

for i in range(len(path_seabass)):
    img = cv2.imread('../../fish_img/seabass/' + path_seabass[i])
    img = cv2.resize(img, (50,50))
    img_seabass.append(img)

X = np.array(img_aji + img_chinu + img_kasago + img_kijihata + img_seabass)
y =  np.array([0]*len(img_aji) + [1]*len(img_chinu) + [2]*len(img_kasago)
 + [3]*len(img_kijihata) + [4]*len(img_seabass))

rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

# データの分割
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ここに解答を記述してください
input_tensor = Input(shape=(50, 50, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

top_model.summary()

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, epochs=50, validation_data=(X_test, y_test))
model.save('model_new.h5')

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

   