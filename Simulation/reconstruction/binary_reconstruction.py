import cv2 as cv
import numpy as np
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import concurrent.futures


def createYRotationMatric(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

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
    img = cv.imread(name) #BGR
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]

    b = Image.fromarray(blue.astype('uint8'), 'L')
    g = Image.fromarray(blue.astype('uint8'), 'L')
    r = Image.fromarray(blue.astype('uint8'), 'L')
    
    b.save("mixed_albedo_blue.jpg")
    g.save("mixed_albedo_green.jpg")
    r.save("mixed_albedo_red.jpg")

    #plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    """
    #plt.imshow(rgb_img, interpolation='bicubic')

    fig = plt.figure() 
    out_img = np.zeros_like(img)
    out_img[:,:,0] = red
    out_img[:,:,1] = red
    out_img[:,:,2] = red

    #fig.add_subplot(1,3,1)
    #plt.imshow(out_img)

    out_img_1 = np.zeros_like(img)
    out_img_1[:,:,0] = green
    out_img_1[:,:,1] = green
    out_img_1[:,:,2] = green

    #fig.add_subplot(1,3,2)
    #plt.imshow(out_img_1)

    out_img_2 = np.zeros_like(img)
    out_img_2[:,:,0] = blue
    out_img_2[:,:,1] = blue
    out_img_2[:,:,2] = blue
    
    #fig.add_subplot(1,3,3)
    #plt.imshow(out_img_2)

    plt.show() 
    """ 
    return

def calculateDiffuseAlbedo(mixed, specular) :
    
    out_img = np.zeros_like(mixed)
    out_img[0] = np.subtract(mixed[0], specular)
    out_img[1] = np.subtract(mixed[1], specular)
    out_img[2] = np.subtract(mixed[2], specular)

    return out_img

def calculateSpecularAlbedo(path, form) :

    prefix = path
    suffix = form

    names = ["x", "x_c", "y", "y_c", "z", "z_c", "b", "w"]

    names = [prefix + name + suffix for name in names]

    images = []
    H, S, V  = [], [], []

    for i in names:
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

    specular_median = np.median(specular_albedo, axis = 0) # median value of rgb.


    ''' This is non-using np-optimization version

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
    '''

    print(specular_median)
    #plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #plt.show()  
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
        encodedImage[i][..., 2] = N_z[..., i] #red normal

        for h in range(height):
            normalize(encodedImage[i][h], copy=False)
            
        # only for visualising
        encodedImage[i] = (encodedImage[i] + 1.0) / 2.0
        encodedImage[i] *= 255.0

        im = Image.fromarray(encodedImage[i].astype('uint8'))
        
        im.save("mixed_normal{}.jpg".format(i))
        
        #fig.add_subplot(1,3,i+1)
        #plt.imshow(im)
    plt.show()
    


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
    R=get_rot_mat(rot_vec,unit=None)

    N = []
    for i in range(3) :
        I_suv_g=np.zeros(images[i].shape)
        I_suv_c=np.zeros(images[i].shape)
        # m, n, 3 x 3, 3 -> m, n, 3
        I_suv_g = np.einsum('mnr,rx->mnx', images[2*i], R)
        I_suv_c = np.einsum('mnr,rx->mnx', images[2*i+1], R)
        print(I_suv_g.shape)
        # pure diffuse component
        G=np.sqrt(I_suv_g[:,:,1]**2 + I_suv_g[:,:,2]**2)
        G_C=np.sqrt(I_suv_c[:,:,1]**2 + I_suv_c[:,:,2]**2)
        print(G - G_C)
        N.append(G-G_C)
        
        
    # for visualising

    encodedImage = np.empty_like(images[0]).astype('float64')
    encodedImage[..., 0] = N[0] # X
    encodedImage[..., 1] = N[1] # Y
    encodedImage[..., 2] = N[2] # Z 

    for h in range(height):
        normalize(encodedImage[h], copy=False)
        
    # only for visualising
    encodedImage = (encodedImage + 1.0) / 2.0
    encodedImage *= 255.0

    im = Image.fromarray(encodedImage.astype('uint8'))
    im.save("diffuse_normal.jpg")
    #plt.imshow(im)
    #plt.show()


if __name__ == "__main__":

    path = "/home/givenone/Desktop/cycle_test_revised_6_hdr/"
    form = ".hdr"

    specular_albedo = calculateSpecularAlbedo(path, form)
    mixed_albedo = calculateMixedAlbedo(path, form)
    diffuse_albedo = calculateDiffuseAlbedo(mixed_albedo, specular_albedo)

    mixed_normal = calculateMixedNormals(path, form)
    diffuse_normal = calculateDiffuseNormals(path, form)

    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #        executor.map(calculateDiffuseNormals, range(1, 11))