from PIL import Image
import numpy as np
import cv2 as cv

def generate_pointcloud(depth, focalLength, sensorSize, cameraLocation) :
    
    # Load Depth Map (Camera Coordinate)
    img = cv.imread(depth, 3)
    arr = np.array(img)
    arr = arr[..., 0]

    height, width = arr.shape

    print(arr.shape)
    centerX = height/2
    centerY = width/2

    pc = [[( (x - centerX) / height * sensorSize * arr[x][y] / focalLength, cameraLocation + arr[x][y], (y - centerY) / width * sensorSize * arr[x][y] / focalLength) for x in range(height)] for y in range(width)]

    np.savetxt("pointcloud.txt", np.reshape(pc, (-1,3)))
    return pc
# focal length = 50mm

generate_pointcloud("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" ,0.005, 0.035, -4.9)