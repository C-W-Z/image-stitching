import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import os
import matplotlib.pyplot as plt
from utils import *


def compute_gradient(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray image
    I = cv2.GaussianBlur(grayImg, (5,5), 0)
    Iy, Ix = np.gradient(I)
    return Ix, Iy

def harris_detector(image, k=0.04, thresRatio=0.01):
    # Compute x and y derivatives of image
    Ix, Iy = compute_gradient(image)

    # Compute products of derivatives at every pixel
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy

    # Compute the sums of the products of derivatives at each pixel
    Sx2 = cv2.GaussianBlur(Ix2, (5,5), 0)
    Sy2 = cv2.GaussianBlur(Iy2, (5,5), 0)
    Sxy = cv2.GaussianBlur(Ixy, (5,5), 0)

    # Compute the response of the detector at each pixel
    """  M = [Sx2 Sxy]
             [Sxy Sy2]  """
    detM = Sx2*Sy2 - Sxy*Sxy
    traceM = Sx2 + Sy2
    R = detM - k*(traceM**2)

    # Threshold on value of R and local maximum
    threshold = thresRatio*np.max(R)
    R[R<threshold] = 0
    localMaxR = maximum_filter(R, size=3, mode='constant')
    R[R<localMaxR] = 0
    # show_heatimage(R)
    point = np.where(R>0)
    point = np.array(point).T  # (2, n) => (n, 2)
    # point[:,[0, 1]] = point[:,[1, 0]]  # (y, x) => (x, y)
    return point

if __name__ == '__main__':
    images, focals = read_images("..\data\parrington\list.txt")
    img1 = images[0]
    img2 = images[1]
    keyPoints1 = harris_detector(img1)
    keyPoints2 = harris_detector(img2)
    print(keyPoints1)
    print(keyPoints2)