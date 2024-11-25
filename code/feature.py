import cv2
import numpy as np
from scipy.spatial import cKDTree
import utils
import harris
import stitch
from enum import IntEnum

class DescriptorType(IntEnum):
    SIFT = 0
    MSOP = 1
    def __str__(self):
        return self.name.upper()

class MotionType(IntEnum):
    TRANSLATION = 0
    AFFINE = 1
    PERSPECTIVE = 2
    def __str__(self):
        return self.name.upper()

def subpixel_refinement(
    gray:np.ndarray[np.uint8, 2],
    keypoints:np.ndarray[int,2]
) -> np.ndarray[float,2]:
    keypoints = keypoints.astype(np.float32)
    keypoints = keypoints.reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    keypoints = cv2.cornerSubPix(gray, cv2.UMat(keypoints), (3, 3), (-1, -1), criteria)
    return keypoints.get().reshape(-1, 2)

def gaussian_blur_with_spacing(
    gray:np.ndarray[np.uint8,2],
    spacing:int=5,
    sigma:float=0
):
    H, W = gray.shape
    newH = H // spacing
    newW = W // spacing

    blurred = cv2.GaussianBlur(gray, (spacing, spacing), sigmaX=sigma, sigmaY=sigma)
    result = cv2.resize(blurred, (newW, newH), interpolation=cv2.INTER_AREA)

    return result

def orientation_histogram(
    patch:np.ndarray[np.uint8,2],
    bins:int=36, margin:int=0,
    centerYX:tuple[float,float]=None
):
    H, W = patch.shape
    assert(H == W)
    patch_size = W - 2 * margin
    # calculate orientations in 12x12 and use 8x8
    ksize = patch_size if patch_size % 2 == 1 else patch_size + 1
    I = cv2.GaussianBlur(patch, (ksize, ksize), sigmaX=4.5, sigmaY=4.5)
    Iy, Ix = np.gradient(I)
    ori = np.mod(np.arctan2(Iy, Ix) * 180 / np.pi, 360)
    ori = ori[margin:margin+patch_size, margin:margin+patch_size]
    if centerYX != None:
        centerYX = (centerYX[0] - margin, centerYX[1] - margin)
        weight = utils.gaussian_weights((patch_size, patch_size), centerYX, 0.15)
        ori *= weight

    # compute major orientation in 8x8 patch
    histogram, *_ = np.histogram(ori, bins, range=(0, 360))
    # major_bin = np.argmax(histogram)
    sorted_bins = np.argsort(histogram)[::-1] # sort from max to min
    major_bin = sorted_bins[0]
    second_bin = sorted_bins[1]

    major_orientation = (major_bin + 0.5) * 360 / bins

    if second_bin >= 0.8 * major_bin:
        second_orientation = (second_bin + 0.5) * 360 / bins
    else:
        second_orientation = None

    return (histogram, major_orientation, second_orientation)

def msop_descriptor(
    gray:np.ndarray[np.uint8, 2],
    keypoints:list[tuple[int, int]]
):
    descriptors = []
    validpoints = []
    orientations = []

    patch_size = 8
    spacing = 5
    sample_size = patch_size * spacing
    max_padding = 2 # min padding = 1
    H, W = gray.shape

    for y, x in keypoints:
        suby, subx = y, x
        y, x = int(np.ceil(y)), int(np.ceil(x))
        _y, _x = suby - y, subx - x

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
        h, w = patch.shape

        # calculate orientations in 12x12 and compute major orientation in 8x8 patch
        _, major_orientation, second_orientation = orientation_histogram(patch, 36, padding)

        def get_desc(orientation:float):
            # get 8x8 orientation patch from 12x12
            rotated = utils.rotate_image(patch, 360 - orientation, (h/2+_x, w/2+_y))
            oriented_patch = rotated[padding:padding+patch_size, padding:padding+patch_size]
            # print(rotated)
            oriented_patch = utils.normalize(oriented_patch).reshape(-1) # 2D to 1D

            validpoints.append((suby, subx))
            descriptors.append(oriented_patch)
            orientations.append(orientation)

        get_desc(major_orientation)
        if second_orientation != None:
            get_desc(second_orientation)

    return (np.array(validpoints), np.array(descriptors), np.array(orientations))

def sift_descriptor(
    gray:np.ndarray[np.uint8, 2],
    keypoints:list[tuple[int, int]]
):
    """
    With SIFT feature detection, gray should be a DoG image.
    Otherwise just use simple gray image.
    """

    validpoints = []
    descriptors = []
    orientations = []

    H, W = gray.shape
    patch_size = 16
    padding = 7
    half = patch_size // 2
    for y, x in keypoints:
        suby, subx = y, x
        y, x = int(np.ceil(y)), int(np.ceil(x))
        _y, _x = suby - y, subx - x

        x_min = x - half - padding
        x_max = x + half + padding
        if x_min < 0 or x_max >= W:
            continue
        y_min = y - half - padding
        y_max = y + half + padding
        if y_min < 0 or y_max >= H:
            continue

        patch = gray[y_min:y_max, x_min:x_max]
        if patch.shape[0] != patch.shape[1]:
            continue

        h, w = patch.shape
        _, major_orientation, second_orientation = orientation_histogram(patch, 36, padding, (h/2+_x, w/2+_y))

        def get_desc(orientation:float):
            # get 16x16 orientation patch from 20x20
            rotated = utils.rotate_image(patch, 360 - orientation, (h/2+_x, w/2+_y))
            oriented_patch = rotated[padding:padding+patch_size, padding:padding+patch_size]
            # print(rotated)
            # assert(oriented_patch.shape == (16, 16))

            # get 4x4x8 desciption vector
            desc = np.zeros((4, 4, 8), dtype=np.uint8)
            for i in range(4):
                for j in range(4):
                    region = oriented_patch[i*4:(i+1)*4, j*4:(j+1)*4]
                    histogram, *_ = orientation_histogram(region, 8, 0)
                    desc[i, j] = histogram
            desc = desc.reshape(-1)

            validpoints.append((suby, subx))
            descriptors.append(desc)
            orientations.append(orientation)

        get_desc(major_orientation)
        if second_orientation != None:
            get_desc(second_orientation)

    for x in range(4):
        for y in range(4):
            patch = gray[x*4:(x+1)*4, y*4:(y+1)*4]

    return np.array(validpoints), np.array(descriptors), np.array(orientations)

def feature_matching(
    descriptors1:np.ndarray[np.uint8,3],
    descriptors2:np.ndarray[np.uint8,3],
    threshold:float
) -> list[tuple[int,int]]:
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
