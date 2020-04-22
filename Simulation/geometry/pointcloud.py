from PIL import Image
import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
import pickle 


def generate_pointcloud(depth, focalLength, cameraLocation, sensor = (0.036, 0.024)) :
    
    # Load Depth Map (Camera Coordinate)
    img = cv.imread(depth, 3)
    arr = np.array(img)
    arr = arr[..., 0]

    height, width = arr.shape
    centerX = height/2
    centerY = width/2

    sensor_width = sensor[0]
    sensor_height = sensor[1]

    pc = [[( (x - centerX) / height * sensor_height * arr[x][y] / focalLength, cameraLocation + arr[x][y], (y - centerY) / width * sensor_width * arr[x][y] / focalLength) for y in range(width)] for x in range(height)]

    with open("pc.txt", "wb") as fp:
        pickle.dump(pc, fp)

    print("Point Cloud Done")
    return pc


def generate_viewing_direction(path, form, focalLength, sensor = (0.036, 0.024)) :

    name = path + "x" + form
    image = cv.imread(name, 3) #BGR
    height, width, _ = image.shape

    centerX = width/2
    centerY = height/2
    sensor_width = sensor[0]
    sensor_height = sensor[1]
    x_pitch = sensor_width/width
    y_pitch = sensor_height/height
    print("pitch: {}, {}".format(x_pitch, y_pitch))

    vd = [ [ ( (float)(x-centerX) * x_pitch,
            (float)(centerY-y) * y_pitch,
            -focalLength)
            for x in range(width)]
            for y in range(height)]

    #vd = [[(0,0,1) for x in range(width)] for y in range(height)]
    v = np.array(vd)
    vd = v.astype('float32')

    # Normalization
    for h in range(height) :
        normalize(vd[h], copy = False)

    print("Viewing Direction Done")
    return vd


#generate_pointcloud("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" , focalLength = 0.005, cameraLocation=-4.9)