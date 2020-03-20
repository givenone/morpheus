import cv2 as cv
import numpy as np
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import concurrent.futures

def calculateMixedNormals():
    images = []

    prefix = "/home/givenone/Desktop/cycle_test_revised_6_hdr/"
    suffix = ".hdr"

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        print(img.shape)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1])# / 255
    N_y = (images[2] - images[3])# / 255
    N_z = (images[4] - images[5])# / 255


    encodedImage = np.empty_like(N_x).astype('float64')

    encodedImage[..., 0] = N_x[..., 2]
    encodedImage[..., 1] = N_y[..., 2]
    encodedImage[..., 2] = N_z[..., 2]

    for h in range(height):
        normalize(encodedImage[h], copy=False)
        #print(encodedImage[h])

    # only for visualising
    encodedImage = (encodedImage + 1.0) / 2.0
    encodedImage *= 255.0

    im = Image.fromarray(encodedImage.astype('uint8'))
    #im.save("mixed.jpg")

    plt.imshow(im)
    plt.show()
    print(img)


def calculateDiffuseNormals(card):

    # for card in range(3, 4):

    images = []

    prefix = "./card{}/".format(card)

    names = [prefix + str(name) + ".TIF" for name in range(3, 16, 2)]
    names.remove(prefix + "11.TIF")

    # print(names)

    for i in names:
        img = Image.open(i)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1]) / 255
    N_y = (images[2] - images[3]) / 255
    N_z = (images[4] - images[5]) / 255

    encodedImage = np.empty_like(N_x).astype('float64')

    encodedImage[..., 0] = N_x[..., 0]
    encodedImage[..., 1] = N_y[..., 0]
    encodedImage[..., 2] = N_z[..., 0]

    for h in range(height):
        normalize(encodedImage[h], copy=False)

    # only for visualising
    encodedImage = (encodedImage + 1.0) / 2.0
    encodedImage *= 255.0

    im = Image.fromarray(encodedImage.astype('uint8'))
    im.save("diffuseNormal{}.png".format(card))



if __name__ == "__main__":


    
    calculateMixedNormals()
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #        executor.map(calculateDiffuseNormals, range(1, 11))