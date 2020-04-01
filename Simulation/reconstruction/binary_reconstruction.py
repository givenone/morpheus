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

date = "0401"

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

def save(path, format, image) :

    cv.imwrite(path+format, image)
    return

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    norm = math.sqrt(np.dot(axis, axis))
    if norm == 0 :
        axis = np.asarray([0,0,0])
    else :
        axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_D(image) : 
    e1 = np.array([1.0, 1.0, 1.0]).astype('float64')
    e1 = e1 / np.linalg.norm(e1)

    height, width, channel = image.shape

    D = np.zeros((height, width)).astype('float64')

    axis = np.cross(image, e1)   # sin theta?  
    D=np.linalg.norm(axis, axis = 2)
    return D
def get_rot_mat(rot_v, unit=None):
    '''
    takes a vector and returns the rotation matrix required to align the unit vector(2nd arg) to it
    '''
    if unit is None:
        unit = [1.0, 0.0, 0.0]

    rot_v = rot_v/np.linalg.norm(rot_v)
    uvw = np.cross(rot_v, unit) #axis of rotation

    rcos = np.dot(rot_v, unit) #cos by dot product
    rsin = np.linalg.norm(uvw) #sin by magnitude of cross product

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw = uvw/rsin
    u, v, w = uvw

    # Compute rotation matrix
    # Artsiom: I haven't checked that this is correct
    R = (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

    return R

    
def calculateMixedAlbedo(path, form) :

    name = path + "w" + form
    img = cv.imread(name, 3) #BGR
    arr = array(img)
    arr = arr.astype('float64')
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    '''
    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]
    '''
    
    print("Mixed Albedo Done")
    #save("output/results" + date + "/mixed_albedo", ".hdr", arr)
    return img # BGR Image



def calculateDiffuseAlbedo(mixed, specular) :
    
    out_img = np.zeros_like(mixed).astype('float64')
    out_img[...,0] = np.subtract(mixed[...,0], specular)
    out_img[...,1] = np.subtract(mixed[...,1], specular)
    out_img[...,2] = np.subtract(mixed[...,2], specular)
    
    print("Diffuse Albedo Done")

    save("output/results" + date + "/diffuse_albedo", ".hdr", out_img)
    return out_img

def calculateSpecularAlbedo(path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    images = []
    H, S, V  = [], [], []

    for i in names:
        # H S V Separation
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float64'))
        
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) #HSV
        h, s, v = cv.split(hsv_img)
        
        arr = array(h)
        H.append(arr.astype('float64'))
        arr = array(s)
        S.append(arr.astype('float64'))
        arr = array(v)
        V.append(arr.astype('float64'))


    height, width, _ = images[0].shape
    
    specular_albedo = [None,None,None]

    for i in range(3) : 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i], V[2*i], H[2*i+1], S[2*i+1], V[2*i+1] 
        
        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        c_g= np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b))
        c_c =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) #chroma
        
        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g, t)

        t = np.divide(c_c, s_g, out=np.zeros_like(c_g), where=s_g!=0)
        spec_c = np.subtract(v_c, t)
        
        specular_albedo[i] = np.maximum(spec, spec_c)

    # TODO :: Fresnel Gain Modulation

    specular_median = np.median(specular_albedo, axis = 0) # median value of rgb.
    cv.imwrite("specular_albedo.hdr", specular_median);
    print("Specular Albedo Done")  

    save("output/results" + date + "/specular_albedo", ".hdr", specular_median)
    return specular_median   


def calculateMixedNormals(path, form):
    images = []

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1])
    N_y = (images[2] - images[3])
    N_z = (images[4] - images[5])

    encodedImage = np.empty_like(N_x).astype('float64')
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

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    for i in names:
        img = cv.imread(i, 3)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    rot_vec = [1,1,1] # specular : white component
    
    N = []
    for i in range(3) :
        """
        I_suv_g=np.zeros(images[i].shape)
        I_suv_c=np.zeros(images[i].shape)
        # m, n, 3 x 3, 3 -> m, n, 3
        I_suv_g = np.einsum('mnr,rx->mnx', images[2*i], R)
        I_suv_c = np.einsum('mnr,rx->mnx', images[2*i+1], R)
    
        # pure diffuse component
        G=np.sqrt(I_suv_g[:,:,1]**2 + I_suv_g[:,:,2]**2)
        G_C=np.sqrt(I_suv_c[:,:,1]**2 + I_suv_c[:,:,2]**2)
        N.append(G-G_C)

        print(images[2*i][850][1700], images[2*i+1][850][1700], I_suv_g[850][1700], I_suv_c[850][1700], (G-G_C)[850][1700])
        """
        """
        R_g = get_inv_mat(images[2*i])
        R_c = get_inv_mat(images[2*i + 1])
        
        I_suv_g = np.einsum("mnx, mnxy -> mny", images[2*i], R_g)
        I_suv_c = np.einsum("mnx, mnxy -> mny", images[2*i+1], R_c)
        G=np.sqrt(I_suv_g[:,:,1]**2 + I_suv_g[:,:,2]**2)
        G_C=np.sqrt(I_suv_c[:,:,1]**2 + I_suv_c[:,:,2]**2)
        """
        G = get_D(images[2*i])
        G_C = get_D(images[2*i + 1])
        N.append(G-G_C)

    encodedImage = np.empty_like(images[0]).astype('float64')
    encodedImage[..., 0] = N[0] # X
    encodedImage[..., 1] = N[1] # Y
    encodedImage[..., 2] = N[2] # Z
    for h in range(height):
        normalize(encodedImage[h], copy=False)  #normalizing

    print("Diffuse Normal Done")  
    plot(encodedImage)
    save("output/results" + date + "/diffuse_normal", ".png", encodedImage)
    return encodedImage

def calculateSpecularNormals(diffuse_albedo, mixed_albedo, mixed_normal, diffuse_normal, viewing_direction) : 
    
    alpha = np.divide(diffuse_albedo[...,0], mixed_albedo[...,0], out=np.zeros_like(mixed_albedo[...,0]), where=mixed_albedo[...,0]!=0)

    d_x = np.multiply(diffuse_normal[..., 0], alpha)
    d_y = np.multiply(diffuse_normal[..., 1], alpha)
    d_z = np.multiply(diffuse_normal[..., 2], alpha)
    alphadiffuse = np.empty_like(diffuse_normal).astype('float64')
    alphadiffuse[..., 0] = d_x
    alphadiffuse[..., 1] = d_y
    alphadiffuse[..., 2] = d_z

    Reflection = np.subtract(mixed_normal, alphadiffuse)
    height, width, _ = Reflection.shape
    
    for h in range(height):
        normalize(Reflection[h], copy=False) # Normalize Reflection

    normal = np.subtract(Reflection, viewing_direction) #subtract ?

    for h in range(height):
        normalize(normal[h], copy=False)
    
    print("Speuclar Normal Done")
    plot(normal)
    save("output/results" + date + "/specular_normal", ".png", normal)
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


    blur = cv.GaussianBlur(normal, (333, 333), 0)
    filtered_normal = cv.subtract(normal, blur)


    print("High Pass Filter Done")
    plot(filtered_normal)
    save("output/results" + date + "/syn_specular_normal", ".png" , filtered_normal)
    return filtered_normal

def synthesize(diffuse_normal, filtered_normal) :
    syn = np.add(diffuse_normal, filtered_normal)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    print("Specular Normal Synthesis done")
    plot(syn)
    return syn

if __name__ == "__main__":

    path = "/home/givenone/Desktop/cycle_test_revised_8_hdr/"
    form = ".hdr"
    '''
    try :
        fp = open("/home/givenone/morpheus/photogeometric/Simulation/geometry/pc.txt", "rb")
        pc = pickle.load(fp)
    except IOError :
        print("Saving pc ...")
        pc = pointcloud.generate_pointcloud("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr"
        , focalLength = 0.005, cameraLocation= (-4.9, 0))

    try :
        print("Saving vd ...") 
        fp = open("/home/givenone/morpheus/photogeometric/Simulation/geometry/vd.txt", "rb")
        vd = pickle.load(fp)
    except IOError : 
        vd = pointcloud.generate_viewing_direction("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" , focalLength = 0.005)
    '''
    vd = pointcloud.generate_viewing_direction("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" , focalLength = 0.005, sensor = (0.025, ))
    specular_albedo = calculateSpecularAlbedo(path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)

    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form)
    specular_normal = calculateSpecularNormals(diffuse_albedo, mixed_albedo, mixed_normal, diffuse_normal, vd)

    filtered_normal = HPF(specular_normal)
    syn = synthesize(diffuse_normal, filtered_normal)