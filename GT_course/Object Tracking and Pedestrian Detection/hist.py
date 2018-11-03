import sys
import cv2
from os.path import dirname
sys.path.append('/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python')
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('/Users/junle/Github/computer_vision/GT_course/Object Tracking and Pedestrian Detection/input_images/pedestrians/000.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])


image = cv2.imread('/Users/junle/Github/computer_vision/GT_course/Object Tracking and Pedestrian Detection/input_images/pedestrians/001.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
