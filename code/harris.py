import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import utils

def edge_feature_filter(
    img:np.ndarray[np.uint8, 3],
    points:list[tuple[float,float]]
):
    H, W, *_ = img.shape
    new_points = []
    for y, x in points:
        y, x = int(y), int(x)
        if x-1 < 0 or x+2 >= W:
            continue
        if y-1 < 0 or y+2 >= H:
            continue
        if np.any(img[y-1:y+2,x-1:x+2, 3] == 0): # alpha = 0
            continue
        new_points.append([y, x])
    return np.array(new_points)

# non-maximum suppression
def nms(
    R:np.ndarray[float,2],
    threshold:float,
    winSize:int
):
    H, W = R.shape
    feature_points = np.empty((0, 2))

    for y in range(0, H, winSize):
        for x in range(0, W, winSize):
            region = R[y:y+winSize, x:x+winSize]
            if np.all(region == region[0,0]):
                continue

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

def harris_detector(
    gray:np.ndarray[float,2],
    sigma:float,
    thresRatio:float=0.01,
    winSize:int=15
):
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
    # points = np.where(R > thresRatio * np.max(R))
    # points = np.array(points).T
    print(f"Find {len(points)} features")
    return points

def multi_scale_harris(
    grays:np.ndarray[float,2],
    sigma:float,
    thres_ratio:float=0.01,
    grid_size:int=20,
    save:bool=False
):
    scales = []

    for i, gray in enumerate(grays):
        points = harris_detector(gray, sigma, thres_ratio, grid_size)
        scales.append(points)
        if save:
            utils.draw_keypoints(gray, points, None, f"multi_scale_harris_{i}")
        grid_size //= 2

    return scales
