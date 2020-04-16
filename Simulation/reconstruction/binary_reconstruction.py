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
        plt.imshow(im, cmap='gray', vmin=0, vmax=100)
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

    v = np.cross(unit_vec, white_vec) # a x b
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

def calculateSpecularAlbedo(path, form) :

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
    
    coefficient = 128 # scale coefficient for albedo 

    for i in range(3) : 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i]/255, V[2*i]/255, H[2*i+1], S[2*i+1]/255, V[2*i+1]/255 
        
        # original albedo separation

        
        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        c_g_1=  np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)) / 255
        c_c_1 =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) / 255 #chroma

        c_g = np.multiply(v_g, s_g)
        c_c = np.multiply(v_c, s_c)

        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g, t) * coefficient #* coefficient
        t = np.divide(c_c, s_g, out=np.zeros_like(c_c), where=s_g!=0)
        spec_c = np.subtract(v_c, t) * coefficient # * coefficient

        mask = v_g > v_c
        # need to mask out error (when saturation is bigger in bright side)
        print(mask[1400][1200], spec[1400][1200], spec_c[1400][1200])
        print(s_g[1400][1200], v_g[1400][1200], c_g[1400][1200], c_g_1[1400][1200])
        print(s_c[1400][1200], v_c[1400][1200], c_c[1400][1200], c_c_1[1400][1200])

        specular_albedo[i] = np.empty_like(v_g)
        specular_albedo[i][mask] = spec[mask]
        specular_albedo[i][~mask] = spec_c[~mask]
        

        #specular_albedo[i][spec < 0] = spec_c[spec < 0]
        #specular_albedo[i][spec_c < 0] = spec[spec_c < 0]
    specular_sum = np.zeros_like(specular_albedo[0])
    specular_cnt = np.zeros_like(specular_albedo[0])
    specular_sum_avg = np.zeros_like(specular_albedo[0])
    for i in range(3) :
        max_V = np.empty_like(V[0])
        flag_V = V[2*i] > V[2*i+1]
        max_V[flag_V] = V[2*i][flag_V] / 255
        max_V[~flag_V] = V[2*i + 1][~flag_V] / 255

        flag = specular_albedo[i] > 0
        specular_sum[flag] += specular_albedo[i][flag]
        specular_sum[~flag] += max_V[~flag] * coefficient

        specular_cnt[flag] += 1
        specular_sum_avg[flag] += specular_albedo[i][flag]


    specular_sum = specular_sum / 3
    specular_median = np.median(specular_albedo, axis = 0) # median value of xyz.
    specular_avg = np.divide(specular_sum_avg, specular_cnt, out=max_V * coefficient, where=specular_cnt!=0)
    plt.title("specular_albedo")
    plot(specular_median)
    plt.title("specular_albedo_using V")
    plot(specular_sum)
    plt.title("specular_albedo_average")
    plot(specular_avg)
    print("Specular Albedo Done")  
    
    return specular_sum


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
    return encodedImage

def calculateDiffuseNormals(path, form):
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
    I = get_D() # rotation matrix    
    N = []

    for i in range(3) :
        I_suv_g = I.dot(images[2*i].reshape([-1,3]).transpose()).transpose().reshape(height,width,3)
        I_suv_c = I.dot(images[2*i+1].reshape([-1,3]).transpose()).transpose().reshape(height,width,3)
        G=np.sqrt(I_suv_g[:,:,1]**2 + I_suv_g[:,:,2]**2)
        G_C=np.sqrt(I_suv_c[:,:,1]**2 + I_suv_c[:,:,2]**2)
        N.append(G-G_C)

    encodedImage = np.empty_like(images[0]).astype('float32')
    encodedImage[..., 0] = N[0] # X
    encodedImage[..., 1] = N[1] # Y
    encodedImage[..., 2] = N[2] # Z

    for h in range(height):
        normalize(encodedImage[h], copy=False)  #normalizing
    print("Diffuse Normal Done")  

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
    #save("C:\\Users\\yeap98\\Desktop\\result\\" + "reflection", ".png", Reflection)

    normal = np.add(Reflection, viewing_direction) #subtract ?
    for h in range(height):
        normalize(normal[h], copy=False)

    print("Speuclar Normal Done")

    return normal


def HPF(normal) : # High Pass Filtering for specular normal reconstruction
    
    height, width, _ = normal.shape

    blur = cv.GaussianBlur(normal, (1001, 1001), 0)
    filtered_normal = cv.subtract(normal, blur)
   
    for h in range(height) :
        normalize(filtered_normal[h], copy=False)

    print("High Pass Filter Done")

    return filtered_normal


def synthesize(diffuse_normal, filtered_normal) :
    coefficient = 0.5 # coefficient for high-pass filtered normal
    syn = np.add(diffuse_normal, filtered_normal * coefficient)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    print("Specular Normal Synthesis done")

    return syn


if __name__ == "__main__":

    #"/home/givenone/Desktop/500ms/"#
    path = "C:\\Users\\yeap98\\Desktop\\lightstage\\morpheus\\rendered_images\\1\\"#"C:\\Users\\yeap98\\Desktop\\lightstage\\morpheus\\rendered_images\\cycle_test_revised_8_hdr\\" # input image path
    form = ".bmp"
    #dist = "/home/givenone/morpheus/photogeometric/Simulation/output/dist_new.hdr" # distance vector path
    
    vd = pointcloud.generate_viewing_direction(path, form, focalLength = 0.1, sensor = (0.0737, 0.0492))
    specular_albedo = calculateSpecularAlbedo(path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)
    
    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form)
    specular_normal = calculateSpecularNormals(diffuse_albedo, specular_albedo, mixed_normal, diffuse_normal, vd)
    filtered_normal = HPF(specular_normal)
    syn = synthesize(diffuse_normal, filtered_normal)
    
    
    plt.title("mixed_normal")
    plot(mixed_normal)
    plt.title("diffuse normal")
    plot(diffuse_normal)
    plt.title("specular_normal")
    plot(specular_normal)

    plt.title("filtered_normal")
    plot(filtered_normal)

    plt.title("Synthesized")
    plot(syn)

    desktop = "C:\\Users\\yeap98\\Desktop\\result\\"
    
    save(desktop + "mixed", ".png", mixed_normal)
    save(desktop + "diffuse", ".png", diffuse_normal)
    save(desktop + "specular", ".png", specular_normal)
    save(desktop + "filtered", ".png", filtered_normal)
    save(desktop + "syn", ".png", syn)
    