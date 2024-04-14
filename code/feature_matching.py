import cv2
import numpy as np
from scipy.spatial import cKDTree
import utils
from Harris_by_ShuoEn import *

def subpixel_refinement(gray:np.ndarray[np.uint8, 2], keypoints:np.ndarray[int,3]):
    return keypoints

def gaussian_blur_with_spacing(gray:np.ndarray[np.uint8,2], spacing:int=5, sigma:float=0):
    H, W = gray.shape
    newH = H // spacing
    newW = W // spacing
    result = np.zeros((newH, newW), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (spacing, spacing), sigmaX=sigma, sigmaY=sigma)

    for i in range(newH):
        for j in range(newW):
            row_start = i * spacing
            row_end = row_start + spacing
            col_start = j * spacing
            col_end = col_start + spacing
            average = np.mean(blurred[row_start:row_end, col_start:col_end])
            result[i, j] = average.astype(np.uint8)

    return result

def orientation_histogram(patch:np.ndarray[np.uint8,2], bins:int=36, margin:int=0):
    H, W = patch.shape
    assert(H == W)
    patch_size = W - 2 * margin
    # calculate orientations in 12x12 and use 8x8
    ksize = patch_size if patch_size % 2 == 1 else patch_size + 1
    I = cv2.GaussianBlur(patch, (ksize, ksize), sigmaX=4.5, sigmaY=4.5)
    Iy, Ix = np.gradient(I)
    ori = np.mod(np.arctan2(Iy, Ix) * 180 / np.pi, 360)
    ori = ori[margin:margin+patch_size, margin:margin+patch_size]

    # compute major orientation in 8x8 patch
    histogram, *_ = np.histogram(ori, bins, range=(0, 360))
    major_bin = np.argmax(histogram)
    major_orientation = (major_bin + 0.5) * 360 / bins

    return (histogram, major_orientation)

def feature_descriptor(gray:np.ndarray[np.uint8, 2], keypoints:list[tuple[int, int]]):
    subpixel_keypoints = subpixel_refinement(gray, keypoints)

    descriptors = []
    validpoints = []
    orientations = []

    patch_size = 8
    spacing = 5
    sample_size = patch_size * spacing
    max_padding = 2 # min padding = 1
    H, W = gray.shape

    for i, (y, x) in enumerate(keypoints):
        # find maximal padding
        half = sample_size // 2
        padding = None
        for p in range(max_padding, 0, -1):
            x_min = x - half - p * spacing
            x_max = x + half + p * spacing
            if x_min < 0 or x_max >= W:
                continue
            y_min = y - half - p * spacing
            y_max = y + half + p * spacing
            if y_min < 0 or y_max >= H:
                continue
            padding = p
            break
        if padding == None:
            continue

        # example: padding = 2
        # sample 12x12 from 60x60
        patch = gaussian_blur_with_spacing(gray[y_min:y_max, x_min:x_max], spacing, 0)

        # calculate orientations in 12x12 and compute major orientation in 8x8 patch
        _, major_orientation = orientation_histogram(patch, 36, padding)

        # get 8x8 orientation patch from 12x12
        rotated = utils.rotate_image(patch, 360 - major_orientation)
        oriented_patch = rotated[padding:padding+patch_size, padding:padding+patch_size]
        # print(rotated)
        oriented_patch = utils.normalize(oriented_patch).reshape(-1) # 2D to 1D

        # sub-pixel refinement ?
        suby, subx = subpixel_keypoints[i]

        validpoints.append((suby, subx))
        descriptors.append(oriented_patch)
        orientations.append(major_orientation)

    return (np.array(validpoints), np.array(descriptors), np.array(orientations))

def feature_matching(descriptors1:np.ndarray[np.uint8,3], descriptors2:np.ndarray[np.uint8,3], threshold:float) -> list[tuple[int,int]]:
    # print(descriptors1.shape, descriptors2.shape)
    # use kd-tree to find two nearest matching points for each decriptors
    tree = cKDTree(descriptors2)
    distances, indices = tree.query(descriptors1, k=2)

    # return the tuple of indices of matches in two descriptors
    # matches = [(i, j), ...] and i, j are indices in descriptors1, descriptors2
    matches = []
    for i, (d1, d2) in enumerate(zip(distances[:, 0], distances[:, 1])):
        # Lowe's ratio test
        if d1 < threshold * d2:
            matches.append((i, indices[i, 0]))
    print("Find Match Features:", len(matches))
    return matches

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")
    H, W, _ = imgs[0].shape
    imgs = imgs[6:8]
    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(len(imgs))]
    keypoints = [harris_detector(img, thresRatio=0.001) for img in imgs]
    descs = []
    points = []
    orientations = []
    for i in range(len(imgs)):
        gray = cv2.cvtColor(projs[i], cv2.COLOR_BGR2GRAY)
        p, d, o = feature_descriptor(gray, keypoints[i])
        points.append(p)
        descs.append(d)
        orientations.append(o)
    matches = feature_matching(descs[0], descs[1], 0.7)
    match_idx1 = np.array([i for i, _ in matches], dtype=np.int32)
    match_idx2 = np.array([j for _, j in matches], dtype=np.int32)
    matched_keypoints1 = points[0][match_idx1]
    orientations1 = orientations[0][match_idx1]
    matched_keypoints2 = points[1][match_idx2]
    orientations2 = orientations[1][match_idx2]
    utils.draw_keypoints(imgs[0], matched_keypoints1, orientations1, "testmatch0.jpg")
    utils.draw_keypoints(imgs[1], matched_keypoints2, orientations2, "testmatch1.jpg")
