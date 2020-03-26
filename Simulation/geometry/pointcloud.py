from PIL import Image
import numpy as np
import cv2 as cv

def generate_pointcloud(depth, focalLength, cameraLocation, sensor = (36, 24)) :
    
    # Load Depth Map (Camera Coordinate)
    img = cv.imread(depth, 3)
    arr = np.array(img)
    arr = arr[..., 0]

    height, width = arr.shape

    print(arr.shape)
    centerX = height/2
    centerY = width/2

    sensor_width = sensor[0]
    sensor_height = sensor[1]

    pc = [[( (x - centerX) / height * sensor_height * arr[x][y] / focalLength, cameraLocation + arr[x][y], (y - centerY) / width * sensor_width * arr[x][y] / focalLength) for x in range(height)] for y in range(width)]

    np.savetxt("pointcloud.txt", np.reshape(pc, (-1,3)))
    return pc
# focal length = 50mm

#generate_pointcloud("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/dist.hdr" , focalLength = 0.005, cameraLocation=-4.9)