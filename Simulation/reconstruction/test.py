import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

normal = cv.imread("diffuse_normal.jpg")
blur = cv.GaussianBlur(normal, (99,99), 0)
filtered_normal = cv.subtract(normal, blur)

plt.imshow(filtered_normal)
plt.show()