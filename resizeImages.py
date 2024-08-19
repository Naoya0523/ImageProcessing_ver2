import cv2
import numpy as np

img = cv2.imread('nearest.jpg')
print(img.shape)
img = img[500:1000,500:1000]
print(img.shape)
cv2.imwrite('images/figure_nearest.jpg', img)