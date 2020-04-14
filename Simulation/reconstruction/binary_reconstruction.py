import cv2 as cv
import numpy as np
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pickle
import math
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import geometry.pointcloud as pointcloud
import concurrent.futures


def plot(image) :
    #Input : np array

    a = image.shape
    height, width = a[0], a[1]

    if(len(a) == 3) :
        for h in range(height):
            normalize(image[h], copy=False)  #normalizing

        image = (image + 1.0) / 2.0
        image *= 255.0
        im = Image.fromarray(image.astype('uint8'))

        plt.imshow(im)
        plt.show()

    else : #gray scale image
        im = Image.fromarray(image.astype('uint8'), 'L')
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.show()

def save(path, form, image) :

    image = (image + 1.0) / 2.0
    image *= 255.0
    im = Image.fromarray(image.astype('uint8'))
    im.save(path+form)
    return


def get_D(unit_vec = None) : # get magnitude of diffuse component in rgb space 
    #height, width, channel = image.shape

    if unit_vec == None :
        unit_vec = [1.0, 0.0, 0.0]
    white_vec = [1.0, 1.0, 1.0]    
    white_vec = white_vec / np.linalg.norm(white_vec)

    v = np.cross(white_vec, unit_vec) # a x b
    s = np.linalg.norm(v) # norm of v
    c = np.dot(white_vec, unit_vec) # a dot b
    u, v, w = v
    v_s = np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ])

    R = (
        np.eye(3) + # I
        v_s +
        v_s.dot(v_s) * (1.0 - c) / (s*s)
    )
    # return matrix * (dot product) rgb space coordinate = suv space coordinate
    return np.linalg.inv(R.transpose())  # return inverse of rotation vecotr (column-wise)
    
def calculateMixedAlbedo(path, form) :

    name = path + "w" + form
    img = cv.imread(name, 3) #BGR
    arr = array(img)
    arr = arr.astype('float32')
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    print("Mixed Albedo Done")

    plt.title("mixed_albedo")
    plt.imshow(rgb_img)
    plt.show()
    return arr # BGR Image



def calculateDiffuseAlbedo(mixed, specular) :
    
    out_img = np.zeros_like(mixed).astype('float32')
    out_img[...,0] = np.subtract(mixed[...,0], specular)
    out_img[...,1] = np.subtract(mixed[...,1], specular)
    out_img[...,2] = np.subtract(mixed[...,2], specular)
    
    print("Diffuse Albedo Done")

    plt.title("diffuse_albedo")
    plt.imshow(cv.cvtColor((out_img).astype('uint8'), cv.COLOR_BGR2RGB))
    plt.show()
    return out_img # BGR

def calculateSpecularAlbedo(viewing_direction, path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y"]

    names = [prefix + name + suffix for name in names]

    images = []
    H, S, V  = [], [], []

    for i in names:
        # H S V Separation
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float32'))
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) #HSV
        h, s, v = cv.split(hsv_img)
        
        arr = array(h)
        H.append(arr.astype('float32'))
        arr = array(s)
        S.append(arr.astype('float32'))
        arr = array(v)
        V.append(arr.astype('float32'))


    height, width, _ = images[0].shape
    
    specular_albedo = [None,None,None] #original
    specular_V = [None,None,None] #original

    v = np.array(viewing_direction)
    v = -v.astype('float32')
    for h in range(height):
        normalize(v[h], copy=False)  #normalizing

    for i in range(3) : 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i], V[2*i], H[2*i+1], S[2*i+1], V[2*i+1] 
        
        # original albedo separation

        coefficient = 128 # scale coefficient for albedo 

        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        c_g=  np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b))
        c_c =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) #chroma


        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g/255, t) * coefficient #* coefficient
        t = np.divide(c_c, s_g, out=np.zeros_like(c_c), where=s_g!=0)
        spec_c = np.subtract(v_c/255, t) * coefficient # * coefficient

        mask = v_g > v_c
        specular_albedo[i] = np.empty_like(v_g)
        specular_albedo[i][mask] = spec[mask]
        specular_albedo[i][~mask] = spec_c[~mask]

        specular_albedo[i][spec < 0] = spec_c[spec < 0]
        specular_albedo[i][spec_c < 0] = spec[spec_c < 0]
                
    specular_median = np.average(specular_albedo, axis = 0) # median value of xyz.
    
    plt.title("specular_albedo")
    plot(specular_median)

    print("Specular Albedo Done")  
    return specular_median


def calculateMixedNormals(path, form):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y"]


    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float32'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1])
    N_y = (images[2] - images[3])
    N_z = (images[4] - images[5])

    encodedImage = np.empty_like(N_x).astype('float32')
    encodedImage[...,0] = N_x[..., 0] #Mixed Normal -> blue component
    encodedImage[...,1] = N_y[..., 0]
    encodedImage[...,2] = N_z[..., 0]

    for h in range(height):
            normalize(encodedImage[h], copy=False)  #normalizing
    
    print("Mixed Normal Done")
    plt.title("mixed normal")
    plot(encodedImage)
    return encodedImage

def calculateDiffuseNormals(path, form, diffuse_albedo):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float32'))

    height, width, _ = images[0].shape

    rot_vec = [1,1,1] # specular : white component
    
    N = []
    for i in range(3) :
        G = get_D(images[2*i], diffuse_albedo)
        G_C = get_D(images[2*i + 1], diffuse_albedo)
        N.append(G-G_C)

    encodedImage = np.empty_like(images[0]).astype('float32')
    encodedImage[..., 0] = N[0] # X
    encodedImage[..., 1] = N[1] # Y
    encodedImage[..., 2] = N[2] # Z

    for h in range(height):
        normalize(encodedImage[h], copy=False)  #normalizing
   
    print("Diffuse Normal Done")  
    plt.title("diffuse normal")
    plot(encodedImage)
    return encodedImage

def calculateSpecularNormals(diffuse_albedo, specular_albedo, mixed_normal, diffuse_normal, viewing_direction) : 
    
    su = specular_albedo + diffuse_albedo[...,0]
    alpha = np.divide(diffuse_albedo[...,0], su, out=np.zeros_like(su), where=su!=0)

    d_x = np.multiply(diffuse_normal[..., 0], alpha)
    d_y = np.multiply(diffuse_normal[..., 1], alpha)
    d_z = np.multiply(diffuse_normal[..., 2], alpha)
    alphadiffuse = np.empty_like(diffuse_normal).astype('float32')
    alphadiffuse[..., 0] = d_x
    alphadiffuse[..., 1] = d_y
    alphadiffuse[..., 2] = d_z

    Reflection = np.subtract(mixed_normal, alphadiffuse)
    height, width, _ = Reflection.shape
    
    for h in range(height):
        normalize(Reflection[h], copy=False) # Normalize Reflection
    plt.title("reflection")
    plot(Reflection)
    normal = np.subtract(Reflection, viewing_direction) #subtract ?

    for h in range(height):
        normalize(normal[h], copy=False)
    
    print("Speuclar Normal Done")
    plt.plot("specular normal")
    plot(normal)
    return normal


def HPF(normal) : # High Pass Filtering for specular normal reconstruction
    
    height, width, _ = normal.shape

    blur = cv.GaussianBlur(normal, (501, 501), 0)
    filtered_normal = cv.subtract(normal, blur)
   
    for h in range(height):
        normalize(filtered_normal[h], copy=False)

    print("High Pass Filter Done")
    plot(filtered_normal)
    return filtered_normal

def synthesize(diffuse_normal, filtered_normal) :
    coefficient = 0.4 # coefficient for high-pass filtered normal
    syn = np.add(diffuse_normal, filtered_normal * coefficient)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    print("Specular Normal Synthesis done")
    plot(syn)
    return syn


if __name__ == "__main__":

    #"/home/givenone/Desktop/500ms/"#
    path = "C:\\Users\\yeap98\\Desktop\\lightstage\\morpheus\\rendered_images\\cycle_test_revised_9_png\\" # input image path
    form = ".png"
    dist = "/home/givenone/morpheus/photogeometric/Simulation/output/dist_new.hdr" # distance vector path
    """
    vd = pointcloud.generate_viewing_direction(dist, focalLength = 0.005, sensor = (0.025, 0.024))
    specular_albedo = calculateSpecularAlbedo(vd, path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)

    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form, diffuse_albedo)
    specular_normal = calculateSpecularNormals(diffuse_albedo, specular_albedo, mixed_normal, diffuse_normal, vd)

    filtered_normal = HPF(specular_normal)
    syn = synthesize(diffuse_normal, filtered_normal)
    """