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
    
from scipy.spatial.transform import Rotation as Rot
def get_D_quat(unit_vec = None) : # get magnitude of diffuse component in rgb space 
    #height, width, channel = image.shape

    if unit_vec == None :
        v1 = [1.0, 0.0, 0.0] # any axis
    else:
        v1 = unit_vec
    v2 = [1.0, 1.0, 1.0]    # white vector
    v2 = v2 / np.linalg.norm(v2)

    rot_axis = np.cross(v1, v2) # a x b
    rot_angle = np.arccos(np.dot(v1, v2))
    cos_val = np.cos(rot_angle/2.0)
    sin_val = np.sin(rot_angle/2.0)
    #print(rot_axis)

    rot1 = Rot.from_quat([rot_axis[0]*sin_val, rot_axis[1]*sin_val, rot_axis[2]*sin_val, cos_val])
    #rot2 = Rot.from_rotvec(rot_angle * rot_axis)

    #print(rot1.as_euler('zyx', degrees=True))
    #print(rot2.as_euler('zyx', degrees=True))
    
  
    return Rot.inv(rot1).as_matrix()
    #return Rot.inv(rot2).as_matrix()
        
def calculateMixedAlbedo(path, form) :

    names = ["x", "x_c", "y", "y_c", "z", "z_c"]
    names = [path + name + form for name in names]

    images = []
    for name in names :
        img = cv.imread(name, 3) #BGR
        arr = array(img).astype('float32')
        images.append(arr)
    
    sum_img = np.zeros_like(images[0])
    for i in images :
        sum_img += i
     
    rgb_img = cv.cvtColor((sum_img/6).astype('uint8'), cv.COLOR_BGR2RGB)

    print("Mixed Albedo Done")

    #plt.title("mixed_albedo")
    #plt.imshow(rgb_img)
    #plt.show()
    return sum_img/3 # BGR Image



def calculateDiffuseAlbedo(mixed, specular) :
    
    out_img = np.zeros_like(mixed).astype('float32')
    out_img[...,0] = np.subtract(mixed[...,0], specular)  
    out_img[...,1] = np.subtract(mixed[...,1], specular) 
    out_img[...,2] = np.subtract(mixed[...,2], specular) 
    
    print("Diffuse Albedo Done")

    #plt.title("diffuse_albedo")
    #plt.imshow(cv.cvtColor((out_img).astype('uint8'), cv.COLOR_BGR2RGB))
    #plt.show()
    return out_img # BGR

def calculateSpecularAlbedo(path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c"]

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

    for i in range(3) : 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i], V[2*i], H[2*i+1], S[2*i+1], V[2*i+1]
        
        # original albedo separation
        
        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        
        # chroma
        c_g =  np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b))
        c_c =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) 

        #c_g = np.multiply(v_g, s_g) -> Same value with range(rgb)
        #c_c = np.multiply(v_c, s_c)
        """
        # another implementation of chroma
        alpha = np.subtract(np.subtract(2*r, g), b) * 0.5
        beta = np.sqrt(3) * np.subtract(g, b) * 0.5
        c_g = np.sqrt(np.add(alpha**2, beta**2)) 

        alpha = np.subtract(np.subtract(2*r_c, g_c), b_c) * 0.5
        beta = np.sqrt(3) * np.subtract(g_c, b_c) * 0.5
        c_c = np.sqrt(np.add(alpha**2, beta**2)) 
        """

        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g, t*128) 
        t = np.divide(c_c, s_g, out=np.zeros_like(c_c), where=s_g!=0)
        spec_c = np.subtract(v_c, t*128) 

        mask = v_g > v_c
        # need to mask out error (when saturation is bigger in bright side)
        specular_albedo[i] = np.empty_like(v_g)
        specular_albedo[i][mask] = spec[mask]
        specular_albedo[i][~mask] = spec_c[~mask]

    """
    # averaging specular values    
    specular_sum = np.zeros_like(specular_albedo[0])
    specular_cnt = np.zeros_like(specular_albedo[0])
    
    for i in range(3) :
        flag = specular_albedo[i] > 0
        specular_sum[flag] += specular_albedo[i][flag]
        specular_cnt[flag] += 1

    specular_average =  np.divide(specular_sum, specular_cnt, out=np.zeros_like(specular_sum), where=specular_cnt!=0)
    plt.title("specular_albedo_using V")
    plot(specular_average)
    """
    
    # using normailzed max values as the specular
    specular_max = np.amax(specular_albedo, axis=0)
    min_val = np.min(specular_max)
    max_val = np.max(specular_max)
    #print("specular_max: min={}, max={}".format(min_val, max_val))
    specular_max = (specular_max - min_val) / (max_val - min_val) * 128.0 # 255 is too bright
    #plt.title("specular_albedo_normalized_max")
    #plot(specular_max) 
    
    print("Specular Albedo Done")  
    
    #return specular_average
    return specular_max


def calculateMixedNormals(path, form):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c"]


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

    names = ["x", "x_c", "y", "y_c", "z", "z_c"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float32'))

    height, width, _ = images[0].shape

    rot_vec = [1,1,1] # specular : white component
    I = get_D_quat() # rotation matrix    
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
    #plt.title("reflection")
    #plot(Reflection)
    #save("C:\\Users\\yeap98\\Desktop\\result\\" + "reflection", ".png", Reflection)

    normal = np.add(Reflection, -viewing_direction).astype('float32')
    for h in range(height):
        normalize(normal[h], copy=False)

    print("Speuclar Normal Done")

    return normal

from scipy.ndimage import gaussian_filter
def HPF(normal) : # High Pass Filtering for specular normal reconstruction
    
    height, width, _ = normal.shape
    
    #blur = cv.GaussianBlur(normal, (17, 17), cv.BORDER_DEFAULT)
    blur = np.zeros_like(normal)
    blur[..., 0] = gaussian_filter(normal[..., 0], sigma=7)
    blur[..., 1] = gaussian_filter(normal[..., 1], sigma=7)
    blur[..., 2] = gaussian_filter(normal[..., 2], sigma=7)
    filtered_normal = cv.subtract(normal, blur)

    
    #plt.title("blur")
    #plot(blur)
    
    #plt.title("filtered")
    #plot(filtered_normal)  
    

    #for h in range(height) :
    #    normalize(filtered_normal[h], copy=False)

    print("High Pass Filter Done")

    return filtered_normal

def HPF_test(diffuse, specular) : # High Pass Filtering for specular normal reconstruction
    
    height, width, _ = specular.shape
    
    filtered_normal = np.subtract(specular, diffuse)

    print("High Pass Filter Done")

    return filtered_normal

def synthesize(diffuse_normal, filtered_normal) :

    syn = np.add(diffuse_normal, filtered_normal)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    print("Specular Normal Synthesis done")

    return syn


if __name__ == "__main__":

    form = ".png"#".bmp"
    path = "C:\\Users\\yeap98\\Desktop\\lightstage\\morpheus\\rendered_images\\cycle_test_revised_9_png\\" 
        
    vd = pointcloud.generate_viewing_direction(path, form, focalLength = 0.05, sensor = (0.025, 0.024))
    #vd = pointcloud.generate_viewing_direction(path, form, focalLength = 0.012, sensor = (0.0689, 0.0492))
    specular_albedo = calculateSpecularAlbedo(path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)
    
    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form)
    specular_normal = calculateSpecularNormals(diffuse_albedo, specular_albedo, mixed_normal, diffuse_normal, vd)
    filtered_normal = HPF(specular_normal)
    #filtered_normal = HPF_test(diffuse_normal, specular_normal)
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

    desktop = "/media/spyo/Workspace/Work/Lightstage/lightstage/simulation & post-processing/results/" #"C:\\Users\\yeap98\\Desktop\\result\\"
    
    save(desktop + "mixed", ".png", mixed_normal)
    save(desktop + "diffuse", ".png", diffuse_normal)
    save(desktop + "specular", ".png", specular_normal)
    save(desktop + "filtered", ".png", filtered_normal)
    save(desktop + "syn", ".png", syn)
    