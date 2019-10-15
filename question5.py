from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import cv2




batch_size = 128
num_classes = 10
epochs = 12


img_rows, img_cols = 28, 28


(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = load_model('model.h5')

# img = load_img('test.bmp')
# img_array = img_to_array(img)
# print(img_array.shape)


from keras.models import load_model
import cv2
import numpy as np


img_test = cv2.imread('numero.bmp', 0)


def separate_numbers_from_img(img):
    transposed = np.transpose(img)
    parsing_img = False
    extracted_numbers = []
    new_img = []
    for col in transposed:
        black_detected = False
        for elem in col:
            # On cherche un pixel qui ne soit pas blanc
            # Si on le trouve on dit qu'on commence une nouvelle image
            if elem != 255:
                if not parsing_img:
                    parsing_img = True
                black_detected = True
                break

        # Si on a dectecte un pixel pas blanc alors on ajoute la colonne à l'image
        if black_detected:
            new_img.append(col)
        # Si aucun pixel noir a ete dectecte,
        else:
        # on est peut etre sorti d'une image
            if parsing_img:
                new_img = np.transpose(new_img)
                extracted_numbers.append(new_img)
                new_img = []
                parsing_img = False

    return extracted_numbers

im = cv2.imread("test.bmp",cv2.IMREAD_GRAYSCALE)
print(im.shape)

im = separate_numbers_from_img(im)

print(len(im))






imlist=[]
for frame in im:
    cv2.imshow("lala",frame)
    cv2.waitKey(0)
    newim=np.resize(frame, (28, 28,1)) 

    img=np.expand_dims(newim, axis=0)
    print(img.shape)
    # imlist.append(img)    
    prediction = model.predict(img)
    print(prediction)
    
    plt.hist(prediction, bins=30)
    plt.ylabel('Probability');
    plt.show()

        


# x=(x_test[0])
# x=np.expand_dims(x, axis=0)
# prediction = model.predict(x)
# print(prediction)


# plt.hist(prediction, bins=30)
# plt.ylabel('Probability');
# plt.show()


# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
