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
        plt.imshow(im, cmap='gray', vmin=0, vmax=30)
        plt.show()

def save(path, form, image) :

    image = (image + 1.0) / 2.0
    image *= 255.0
    im = Image.fromarray(image.astype('uint8'))
    im.save(path+form)
    return


def get_D(image, diffuse_albedo) : 
    height, width, channel = image.shape

    z = np.copy(diffuse_albedo)
    for h in range(height):
        normalize(diffuse_albedo[h], copy=False)  #normalizing

    D = np.zeros((height, width)).astype('float32')
    D = np.einsum('abx, abx -> ab', image, diffuse_albedo)

    return D

    
def calculateMixedAlbedo(path, form) :

    name = path + "w" + form
    img = cv.imread(name, 3) #BGR
    arr = array(img)
    arr = arr.astype('float32')
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]
    print("Mixed Albedo Done")

    plt.title("mixed_albedo")
    plt.imshow(rgb_img)
    plt.show()
    #cv.imwrite('mixed_albedo.hdr', blue)
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
    #cv.imwrite('diffuse_albedo.hdr', out_img)
    return out_img # BGR

def calculateSpecularAlbedo(viewing_direction, path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y", "b", "w"]

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
    specular_albedo_frsenel = [None,None,None] #frsenel modulated

    light = [ (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    v = np.array(viewing_direction)
    v = -v.astype('float32')
    for h in range(height):
        normalize(v[h], copy=False)  #normalizing

    for i in range(3) : 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i], V[2*i], H[2*i+1], S[2*i+1], V[2*i+1] 
        
        # original albedo separation

        coefficient = 30 # scale coefficient for albedo 

        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        c_g=  np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b))
        c_c =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) #chroma

        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g/100, t) * coefficient #* coefficient
        t = np.divide(c_c, s_g, out=np.zeros_like(c_c), where=s_g!=0)
        spec_c = np.subtract(v_c/100, t) * coefficient # * coefficient
        
        specular_albedo[i] = np.maximum(spec, spec_c)
        

        # fresnel
        
        fresnel = np.full( (height, width, 3), light[2*i])
        fresnel = fresnel.astype('float32')

        fresnel_c = np.full( (height, width, 3), light[2*i+1])
        fresnel_c = fresnel_c.astype('float32')
        
        E = v + fresnel
        for h in range(height):
            normalize(E[h], copy=False)  #normalizing

        E_c = v + fresnel_c
        for h in range(height):
            normalize(E_c[h], copy=False)  #normalizing

        cos_g = np.einsum('abx, abx -> ab', v, E)
        cos_c = np.einsum('abx, abx -> ab', v, E_c)  
        
        f = np.full( (height,width), 1)
        f = f.astype('float32')
        f = (f - cos_g)
        
        f_c = np.full( (height,width), 1)
        f_c = f_c.astype('float32')
        f_c = (f_c - cos_c)
        
        f_spec = spec * f
        f_spec_c = spec_c * f_c
        #spec = spec + (1.0 - spec) * f
        #spec_c = spec_c + (1.0 - spec_c) * f_c
        specular_albedo_frsenel[i] = np.maximum(f_spec, f_spec_c)

        
    specular_median = np.median(specular_albedo, axis = 0) # median value of xyz.
    specular_median_fresnel = np.median(specular_albedo_frsenel, axis = 0) # median value of xyz.

    plt.title("specular_albedo")
    plot(specular_median)
    plt.title("specular_albedo_fresnel")
    plot(specular_median_fresnel)
    #cv.imwrite('specular_albedo.png', specular_median)
    #cv.imwrite('specular_albedo_fresnel.png', specular_median_fresnel)
    
    print("Specular Albedo Done")  
    return specular_median, specular_median_fresnel


def calculateMixedNormals(path, form):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y", "b", "w"]


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
    return encodedImage

def calculateDiffuseNormals(path, form, diffuse_albedo):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "z", "z_c", "y_c", "y", "b", "w"]

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
    kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                         [-1, -1, -1]])
    
    kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
    
    filtered_normal = cv.filter2D(normal, -1, kernel)
    
    height, width, _ = normal.shape

    for h in range(height):
        normalize(filtered_normal[h], copy=False)


    blur = cv.GaussianBlur(normal, (501, 501), 0)
    filtered_normal = cv.subtract(normal, blur)
   

    print("High Pass Filter Done")
    plot(filtered_normal)
    return filtered_normal

def synthesize(diffuse_normal, filtered_normal) :
    syn = np.add(diffuse_normal, filtered_normal * 0.4)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    print("Specular Normal Synthesis done")
    plot(syn)
    return syn

if __name__ == "__main__":

    path = "/home/givenone/morpheus/photogeometric/rendered_images/cycle_test_revised_9_png/"
    form = ".png"

    vd = pointcloud.generate_viewing_direction("/home/givenone/morpheus/photogeometric/Simulation/output/dist_new.hdr" , focalLength = 0.005, sensor = (0.025, 0.024))
    specular_albedo, specular_albedo_fresnel = calculateSpecularAlbedo(vd, path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)

    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form, diffuse_albedo)
    specular_normal = calculateSpecularNormals(diffuse_albedo, specular_albedo_fresnel, mixed_normal, diffuse_normal, vd)

    filtered_normal = HPF(specular_normal)
    syn = synthesize(diffuse_normal, filtered_normal)