import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import utils

# non-maximum suppression
def nms(R:np.ndarray[float,2], threshold:float, winSize:int):
    H, W = R.shape
    feature_points = np.empty((0, 2))

    for y in range(0, H, winSize):
        for x in range(0, W, winSize):
            region = R[y:y+winSize, x:x+winSize]

            above_threshold = np.where(region > threshold)

            if len(above_threshold[0]) > 0:
                new_y = y + above_threshold[0]
                new_x = x + above_threshold[1]
                new_points = np.array([new_y, new_x]).T
                feature_points = np.concatenate((feature_points, new_points), axis=0)
            else:
                max_index = np.unravel_index(np.argmax(region), region.shape)
                new_y = y + max_index[0]
                new_x = x + max_index[1]
                feature_points = np.append(feature_points, np.array([[new_y, new_x]]), axis=0)

    return feature_points

def harris_detector(gray:np.ndarray[float,2], sigma:float, thresRatio:float=0.01, winSize:int=15):
    k = 0.04
    ksize = (5,5)

    I = cv2.GaussianBlur(gray, ksize, sigmaX=sigma, sigmaY=sigma) 
    Iy, Ix = np.gradient(I)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Sx2 = cv2.GaussianBlur(Ix2, ksize, sigmaX=sigma, sigmaY=sigma)
    Sy2 = cv2.GaussianBlur(Iy2, ksize, sigmaX=sigma, sigmaY=sigma)
    Sxy = cv2.GaussianBlur(Ixy, ksize, sigmaX=sigma, sigmaY=sigma)

    M_Determine = Sx2 * Sy2 - Sxy * Sxy
    M_trace = Sx2 + Sy2
    R = M_Determine - k * (M_trace * M_trace)

    localMaxR = maximum_filter(R, size=3, mode='constant', cval=0)
    R[R<localMaxR] = 0
    points = nms(R, thresRatio * np.max(R), winSize)
    print("Find", len(points), "features")
    return points

def multi_scale_harris(grays:np.ndarray[float,2], sigma:float, thresRatio:float=0.01, winSize:int=15, save:bool=False):
    scales = []

    for i, gray in enumerate(grays):
        points = harris_detector(gray, sigma, thresRatio, winSize)
        scales.append(points)
        if save:
            utils.draw_keypoints(gray, points, None, f"multi_scale_harris_{i}")
        winSize //= 2

    return scales

if __name__ == '__main__':
    images, focals = utils.read_images("data\grail\list.txt")
    img1 = images[1]
    grayImg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sigma_scale = 1

    grays = utils.to_multi_scale(grayImg, 2, 1)
    keypoints = multi_scale_harris(grays, 0.5, thresRatio=0.01, winSize=20, save=True)
