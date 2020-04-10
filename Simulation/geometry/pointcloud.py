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


def generate_viewing_direction(depth, focalLength, sensor = (0.036, 0.024)) :
    
    img = cv.imread(depth, 3)
    arr = np.array(img)
    arr = arr[..., 0]

    height, width = arr.shape

    centerX = width/2
    centerY = height/2

    sensor_width = sensor[0]
    sensor_height = sensor[1]
    vd = [ [   ( (((float)(x- centerX) / (float) (width))) * sensor_width * (arr[y][x]/ focalLength), 
                 (((float)(centerY - y) / (float) (height))) * sensor_height * (arr[y][x]/ focalLength),
                 -arr[y][x])
                for x in range(width)
            ] for y in range(height)
    ]
    #vd = [[( (x - centerX) / height * sensor_height * arr[x][y] / focalLength, arr[x][y], (y - centerY) / width * sensor_width * arr[x][y] / focalLength) for y in range(width)] for x in range(height)]
    
    v = np.array(vd)
    vd = v.astype('float32')

    # Normalization
    for h in range(height) :
        normalize(v[h], copy = False)

    #with open("vd.txt", "wb") as fp:
    #    pickle.dump(vd, fp)

    print("Viewing Direction Done")
    return v


#generate_pointcloud("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" , focalLength = 0.005, cameraLocation=-4.9)