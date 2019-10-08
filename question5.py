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


extracted_numbers =separate_numbers_from_img(img_test)

model = load_model('model.h5')

