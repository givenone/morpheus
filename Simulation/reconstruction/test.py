import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
image = cv.imread("/home/givenone/morpheus/photogeometric/Simulation/reconstruction/syn.tif", 3)
print(image[2000][1000])
image = (image + 1.0) / 2.0
image *= 255.0
im = Image.fromarray(image.astype('uint8'))

plt.imshow(im)
plt.show()