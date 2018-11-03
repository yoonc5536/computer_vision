import cv2
import numpy as np

def checkHSVColor(img_path,bgr_color = False):
    if bgr_color is False:
        img_in = cv2.imread(img_path)
        color = [[img_in[0][0]]]
        bgr_color = np.uint8(color)
        hsv_color = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
    else:
        hsv_color = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
    print('BGR color:{}'.format(bgr_color))
    print('HSV color:{}'.format(hsv_color))
    return hsv_color

img_path = ['./sign_color_samples/stop_sign_sample.png', './sign_color_samples/yield_sign_sample.png',
     './sign_color_samples/dne_sign_sample.png','./sign_color_samples/warning_sign_sample.png','./sign_color_samples/construction_sign_yellow_sample.png',
     './sign_color_samples/yield_centroid_sample.png','./sign_color_samples/dne_centroid_sample.png',
     './sign_color_samples/black_sample.png','./sign_color_samples/traffic_light_gray_area__sample.png']

for path in img_path:
    print(path)
    checkHSVColor(path)