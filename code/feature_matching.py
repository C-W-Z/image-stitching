import cv2
import numpy as np
from scipy.spatial import cKDTree
import utils
from Harris_by_ShuoEn import *
import stitch

def subpixel_refinement(gray:np.ndarray[np.uint8, 2], keypoints:np.ndarray[int,2]) -> np.ndarray[float,2]:
    keypoints = keypoints.astype(np.float32)
    keypoints = keypoints.reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    keypoints = cv2.cornerSubPix(gray, cv2.UMat(keypoints), (3, 3), (-1, -1), criteria)
    return keypoints.get().reshape(-1, 2)

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

def orientation_histogram(patch:np.ndarray[np.uint8,2], bins:int=36, margin:int=0, centerYX:tuple[float,float]=None):
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

def msop_descriptor(gray:np.ndarray[np.uint8, 2], keypoints:list[tuple[int, int]]):
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

def sift_descriptor(gray:np.ndarray[np.uint8, 2], keypoints:list[tuple[int, int]]):
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

    return (np.array(validpoints), np.array(descriptors), np.array(orientations))

def feature_matching(descriptors1:np.ndarray[np.uint8,3], descriptors2:np.ndarray[np.uint8,3], threshold:float) -> list[tuple[int,int]]:
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
    # H, W, _ = imgs[0].shape
    imgs = imgs[1:3]
    focals = focals[1:3]
    N = len(imgs)
    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(len(imgs))]
    # projs = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2BGRA) for i in range(len(imgs))]
    grays = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in projs]
    keypoints = [harris_detector(img, thresRatio=0.05) for img in projs]
    print("Complete Harris Detection")

    descs = []
    points = []
    orientations = []
    for i in range(len(projs)):
        subpixel_keypoints = subpixel_refinement(grays[i], keypoints[i])
        p, d, o = sift_descriptor(grays[i], subpixel_keypoints)
        points.append(p)
        descs.append(d)
        orientations.append(o)
        print("Complete Feature Description:", len(p))
        utils.draw_keypoints(projs[i], keypoints[i], None, f"test__{i}")

    matches = feature_matching(descs[0], descs[1], 0.85)

    match_idx1 = np.array([i for i, _ in matches], dtype=np.int32)
    match_idx2 = np.array([j for _, j in matches], dtype=np.int32)
    matched_keypoints1 = points[0][match_idx1]
    orientations1 = orientations[0][match_idx1]
    matched_keypoints2 = points[1][match_idx2]
    orientations2 = orientations[1][match_idx2]
    utils.draw_keypoints(projs[0], matched_keypoints1, orientations1, "testmatch0.jpg")
    utils.draw_keypoints(projs[1], matched_keypoints2, orientations2, "testmatch1.jpg")

    M = stitch.ransac_homography(matched_keypoints1, matched_keypoints2, 1, 5000)
    # M, _ = cv2.findHomography(matched_keypoints1[:, ::-1], matched_keypoints2[:, ::-1], cv2.RANSAC, ransacReprojThreshold=0.01, confidence=0.99)
    # M, _ = cv2.estimateAffinePartial2D(matched_keypoints1[:, ::-1], matched_keypoints2[:, ::-1], method=cv2.RANSAC, ransacReprojThreshold=1 ,confidence=0.999)
    print(M)
    # M = [[1, 0, 248.9],
    #      [0, 1, 4.17],
    #      [0, 0, 1]]
    # M = np.array(M)
    result = stitch.stitch_homography(projs[0], projs[1], M)
    cv2.imwrite("test.png", result)
