from keras.models import load_model
import cv2
import numpy as np


img_test = cv2.imread('numero.bmp', 0)
transposed = np.transpose(img_test)

parsing_img = False
img_tab = []
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

    #Si on a dectecte un pixel pas blanc alors on ajoute la colonne à l'image
    if black_detected:
        new_img.append(col)
    #Si aucun pixel noir a ete dectecte,
    else:
        # on est peut etre sorti d'une image
        if parsing_img:
            new_img = np.transpose(new_img)
            img_tab.append(new_img)
            new_img = []
            parsing_img = False










model = load_model('model.h5')

