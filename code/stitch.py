import cv2
import numpy as np
import utils
from feature_matching import *
from Harris_by_ShuoEn import *

def ransac(offsets:np.ndarray[float,3], threshold:float, iterations:int=1000):
    # offsets = keypoints1 - keypoints2
    N = len(offsets)
    best_offset = None
    max_inliner_count = -1
    for _ in range(iterations):
        i = np.random.randint(0, N)
        inliner_count = 0
        for j in range(N):
            if i == j:
                continue
            dy, dx = offsets[i] - offsets[j]
            if np.sqrt(dy ** 2 + dx ** 2) < threshold:
                inliner_count += 1
        if max_inliner_count < inliner_count:
            max_inliner_count = inliner_count
            best_offset = offsets[i]

    print(best_offset)

    # calculate average offset
    totalY = 0
    totalX = 0
    count = 0
    for i in range(N):
        dy, dx = best_offset - offsets[i]
        if np.sqrt(dy ** 2 + dx ** 2) < threshold:
            totalY += offsets[i][0]
            totalX += offsets[i][1]
            count += 1

    return (totalY / count, totalX / count)

def stitch(img_left, img_right, offset):
    dy, dx = offset # dx must be >= 0, img_right needs to be translated right

    new_H = max(img_left.shape[0], int(img_right.shape[0] + np.ceil(dy)))
    new_W = max(img_left.shape[1], int(img_right.shape[1] + np.ceil(dx)))

    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    if dy >= 0:
        combined_image = cv2.warpAffine(img_right, M, (new_W, new_H))
        mask = np.where(img_left != [0, 0, 0])
        combined_image[mask] = img_left[mask]
    else:
        combined_image = np.zeros((new_H, new_W, 3), dtype=np.uint8)
        combined_image[-dy:] = img_right
        combined_image = cv2.warpAffine(img_right, M, (new_W, new_H))
        mask = np.where(img_left != [0, 0, 0])
        translated_mask = [(y + dy, x) for y, x in mask]
        combined_image[mask] = img_left[mask]

    cv2.imwrite("test_stitch.jpg", combined_image)

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
        p, d, o = feature_descriptor(projs[i], keypoints[i])
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
    # utils.draw_keypoints(imgs[0], matched_keypoints1, orientations1, "testmatch0.jpg")
    # utils.draw_keypoints(imgs[1], matched_keypoints2, orientations2, "testmatch1.jpg")

    offsets = matched_keypoints2 - matched_keypoints1
    offset = ransac(offsets, 1, 1000)
    print(offset)
    stitch(projs[1], projs[0], offset)
