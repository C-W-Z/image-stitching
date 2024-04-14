from Harris_by_ShuoEn import *
import numpy as np

def NMS_Harris(image, Point_limit:int, k=0.04, thresRatio=0.01):
    #Point_limit(N): the function will stop after finding N points using NMS
    #thresRaito: Used for determine the Threshold for NMS that knows which feature points to delete

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
    #  Also used for NMS's deletion, which is, when the points are starting to get smaller than the threshold,
    # the function return and exit
    threshold = thresRatio*np.max(R)
    R[R<threshold] = 0
    localMaxR = maximum_filter(R, size=3, mode='constant')
    R[R<localMaxR] = 0

    #return point
    # #point is a list of Feature Points