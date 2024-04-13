import cv2
import numpy as np
import utils
from Harris_by_ShuoEn import *

def feature_descriptor(image:np.ndarray[np.uint8, 3], keypoints:list[tuple[int, int]], bins:int=36, save_filename=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray image
    I = cv2.GaussianBlur(gray, (5,5), sigmaX=4.5, sigmaY=4.5)
    Iy, Ix = np.gradient(I)
    theta = np.arctan2(Iy, Ix) * 180 / np.pi
    theta = np.mod(theta, 360)

    descriptors = []
    validpoints = []
    orientations = []

    patch_size = 16
    H, W, *_ = image.shape
    print(H, W)
    for y, x in keypoints:
        half_patch_size = patch_size // 2
        if x - half_patch_size < 0 or x + half_patch_size >= W:
            continue
        if y - half_patch_size < 0 or y + half_patch_size >= H:
            continue
        # sample 8x8 from 40x40
        x_min = x - half_patch_size
        x_max = x + half_patch_size
        y_min = y - half_patch_size
        y_max = y + half_patch_size
        print("keypoint:", y, x)
        print(y_min, y_max, x_min, x_max)

        patch = theta[y_min:y_max, x_min:x_max]
        # patch = cv2.resize(patch, (8, 8))

        # compute major orientation
        binSize = 360 / bins
        histogram = np.zeros(bins)
        for b in range(bins):
            histogram[b] = np.sum((patch >= b * binSize) & (patch < (b + 1) * binSize))
        major_bin = np.argmax(histogram)
        major_orientation = (major_bin + 0.5) * binSize
        print("major orientation=", major_orientation)
        orientations.append(major_orientation)

        # sub-pixel refinement ?

        rotated = utils.rotate_image(gray, major_orientation, (float(x), float(y)))
        oriented_patch = rotated[y_min:y_max, x_min:x_max]
        # print(oriented_patch)
        oriented_patch = utils.normalize(oriented_patch)

        descriptors.append(((y, x), oriented_patch))
        
        validpoints.append((y, x))

    if save_filename != None:
        utils.draw_keypoints(image, validpoints, orientations, save_filename)

    return descriptors

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")
    H, W, _ = imgs[0].shape
    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(len(imgs))]
    keyPoints = [harris_detector(img) for img in imgs]
    for i in range(len(imgs)):
        desc = feature_descriptor(projs[i], keyPoints[i], save_filename=f"test{i}")
        # print(desc)

