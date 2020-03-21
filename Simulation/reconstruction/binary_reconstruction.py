import cv2 as cv
import numpy as np
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import concurrent.futures

def calculateMixedAlbedo(path, form) :

    name = path + "w" + form
    img = cv.imread(name) #BGR
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #print(img.shape, img[1000, 2080])
    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]

    #plt.imshow(rgb_img, interpolation='bicubic')

    fig = plt.figure() 
    out_img = np.zeros_like(img)
    out_img[:,:,0] = red
    out_img[:,:,1] = red
    out_img[:,:,2] = red
    fig.add_subplot(1,3,1)
    plt.imshow(out_img)

    out_img_1 = np.zeros_like(img)
    out_img_1[:,:,0] = green
    out_img_1[:,:,1] = green
    out_img_1[:,:,2] = green
    fig.add_subplot(1,3,2)
    plt.imshow(out_img_1)

    out_img_2 = np.zeros_like(img)
    out_img_2[:,:,0] = blue
    out_img_2[:,:,1] = blue
    out_img_2[:,:,2] = blue
    
    fig.add_subplot(1,3,3)
    plt.imshow(out_img_2)

    plt.show()  
    return

def calculateDiffuseAlbedo() :

    return

def calculateSpecularAlbedo(path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    bgr_images = []
    h, s, v  = []

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        bgr_images.append(arr.astype('float64'))
        
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) #HSV
        h, s, v = cv2.split(hsv_image)
        arr = array(hsv_img)
        hsv_images.append(arr.astype('float64'))

    height, width, _ = hsv_images[0].shape
    
    specular_albedo = [None,None,None]
    for i in range(3) :
        specular_albedo[i]= np.zeros((height, width))
        a, a_c = i, i+1
        for h in range(height) :
            for w in range(width) :
                chroma_g = max(bgr_images[a][h][w]) - min(bgr_images[a][h][w])
                value_g = hsv_images[a][h][w][2]
                saturation_c = hsv_images[a_c][h][w][1]

                chroma_c = max(bgr_images[a_c][h][w]) - min(bgr_images[a_c][h][w])
                value_c = hsv_images[a_c][h][w][2]
                saturation_g = hsv_images[a][h][w][1]

                delta = value_g - chroma_g/saturation_c if saturation_c != 0 else 0  
                delta_c = value_c - chroma_c/saturation_g if saturation_g != 0 else 0

                specular_albedo[i][h][w] = max(delta, delta_c)

    np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
    np.subtract()

    im = Image.fromarray(specular_albedo[0].astype('uint8'), 'L')

    plt.imshow(im)
    plt.show()  
    return    


def calculateMixedNormals(path, form):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        print(img)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1])# / 255
    N_y = (images[2] - images[3])# / 255
    N_z = (images[4] - images[5])# / 255


    encodedImage = []

    fig = plt.figure() 
    for i in range(3) :
        encodedImage.append(np.empty_like(N_x).astype('float64'))
        encodedImage[i][..., 0] = N_x[..., i]
        encodedImage[i][..., 1] = N_y[..., i]
        encodedImage[i][..., 2] = N_z[..., i] #blue normal

        for h in range(height):
            normalize(encodedImage[i][h], copy=False)
            
        # only for visualising
        encodedImage[i] = (encodedImage[i] + 1.0) / 2.0
        encodedImage[i] *= 255.0

        im = Image.fromarray(encodedImage[i].astype('uint8'))
    #im.save("mixed.jpg")
        fig.add_subplot(1,3,i+1)
        plt.imshow(im)
    plt.show()
    


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

    calculateSpecularAlbedo("/home/givenone/Desktop/cycle_test_revised_5/", ".png")
    #calculateMixedAlbedo("/home/givenone/Desktop/cycle_test_revised_5/", ".png")
    #calculateMixedNormals("/home/givenone/Desktop/cycle_test_revised_6_hdr/", ".hdr")
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #        executor.map(calculateDiffuseNormals, range(1, 11))