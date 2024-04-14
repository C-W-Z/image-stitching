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
            if np.sqrt(dy ** 2 + dx ** 2) <= threshold:
                inliner_count += 1
        if max_inliner_count < inliner_count:
            max_inliner_count = inliner_count
            best_offset = offsets[i]

    # print(best_offset)

    # calculate average offset
    totalY = 0
    totalX = 0
    count = 0
    for i in range(N):
        dy, dx = best_offset - offsets[i]
        if np.sqrt(dy ** 2 + dx ** 2) <= threshold:
            totalY += offsets[i][0]
            totalX += offsets[i][1]
            count += 1

    return (totalY / count, totalX / count)

def stitch(img_left:np.ndarray[np.uint8,3], img_right:np.ndarray[np.uint8,3], offset:tuple[float,float]):
    """
    Parameters
    img_left: the image should be stitch on the left (4 channels BGRA)
    img_right: the image should be stitch on the right (4 channels BGRA)
    offset: (dy, dx), the top-left corner of img_right will be stitched at (dy, dx), dx must >= 0
    """

    dy, dx = offset
    assert(dx >= 0)
    HL, WL = img_left.shape[:2]
    HR, WR = img_right.shape[:2]

    new_W = max(WL, WR + int(np.ceil(dx)))

    if dy >= 0:
        new_H = max(HL, HR + int(np.ceil(abs(dy))))

        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        warped_right = cv2.warpAffine(img_right, M, (new_W, new_H))
        combined = np.zeros((new_H, new_W, 4), dtype=np.uint8)
        mask = np.where(img_left[:, :, 3] > 127) # alpha > 127
        combined[mask] = img_left[mask]

    else:
        new_H = max(HL, HR) + int(np.ceil(abs(dy)))

        M = np.float32([[1, 0, dx],
                        [0, 1,  0]])
        warped_right = cv2.warpAffine(img_right, M, (new_W, new_H))
        M = np.float32([[1, 0,  0],
                        [0, 1, -dy]])
        combined = cv2.warpAffine(img_left, M, (new_W, new_H))

    warped_right[warped_right[:, :, 3] < 225] = 0
    combined[combined[:, :, 3] < 225] = 0

    # Blending
    overlap_x = int(np.floor(dx))

    right_notlap = np.where(
        np.logical_and(
            combined[:, overlap_x:WL, 3] < 128,
            warped_right[:, overlap_x:WL, 3] > 127
        )
    )

    true_overlap = np.where(
        np.logical_and(
            combined[:, overlap_x:WL, 3] > 127,
            warped_right[:, overlap_x:WL, 3] > 127
        )
    )

    overlap_weights = np.linspace(0, 1, WL - overlap_x)
    expanded_weights = np.tile(overlap_weights.reshape(1, -1, 1), (new_H, 1, 4))

    combined[true_overlap[0], true_overlap[1] + overlap_x] = (
        combined[true_overlap[0], true_overlap[1] + overlap_x] * (1 - expanded_weights[true_overlap]) + 
        warped_right[true_overlap[0], true_overlap[1] + overlap_x] * expanded_weights[true_overlap]
    )

    combined[right_notlap[0], right_notlap[1] + overlap_x] = warped_right[right_notlap[0], right_notlap[1] + overlap_x]

    combined[:, WL:] = warped_right[:, WL:]

    return combined

def stitch_all(images, offsets):
    N = len(images)
    assert(len(offset) == N - 1)
    for i, offset in enumerate(offsets):
        pass

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")

    N = len(imgs)
    H, W, _ = imgs[0].shape

    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]
    print("Complete Cylindrical Projection")
    # keypoints = [harris_detector(img, thresRatio=0.001) for img in imgs]
    # print("Complete Harris Detection")

    # descs = []
    # points = []
    # orientations = []
    # for i in range(N):
    #     gray = cv2.cvtColor(projs[i], cv2.COLOR_BGR2GRAY)
    #     p, d, o = feature_descriptor(gray, keypoints[i])
    #     points.append(p)
    #     descs.append(d)
    #     orientations.append(o)
    #     print("Complete Feature Description", i)

    # offsets = []
    # for i in range(N - 1):
    #     matches = feature_matching(descs[i], descs[i + 1], 0.7)
    #     match_idx1 = np.array([i for i, _ in matches], dtype=np.int32)
    #     match_idx2 = np.array([j for _, j in matches], dtype=np.int32)
    #     matched_keypoints1 = points[i][match_idx1]
    #     orientations1 = orientations[i][match_idx1]
    #     matched_keypoints2 = points[i + 1][match_idx2]
    #     orientations2 = orientations[i + 1][match_idx2]
    #     # utils.draw_keypoints(imgs[0], matched_keypoints1, orientations1, "testmatch0.jpg")
    #     # utils.draw_keypoints(imgs[1], matched_keypoints2, orientations2, "testmatch1.jpg")

    #     # left image - right image
    #     # the keypoints are at the right part of left image and left part of right image
    #     sample_offsets = matched_keypoints1 - matched_keypoints2
    #     offset = ransac(sample_offsets, 1, 1000)
    #     print(offset)
    #     # stitch(projs[1], projs[0], offset)
    #     offsets.append(offset)

    # print(offsets)
    offsets = [(5.961538461538462, 251.92307692307693), (3.923076923076923, 241.96153846153845), (3.8484848484848486, 243.96969696969697), (3.888888888888889, 249.05555555555554), (4.171428571428572, 239.11428571428573), (4.918918918918919, 246.1891891891892), (3.8518518518518516, 248.03703703703704), (4.0344827586206895, 240.06896551724137), (4.157894736842105, 245.8421052631579), (4.92, 246.96), (4.043478260869565, 240.95652173913044), (5.0, 250.07692307692307), (4.903225806451613, 241.16129032258064), (5.0, 249.0), (5.888888888888889, 240.61111111111111), (3.9523809523809526, 243.04761904761904), (4.3, 248.7)]

    s = projs[0]
    oy, ox = 0, 0
    for i, offset in enumerate(offsets):
        oy += offsets[i][0]
        ox += offsets[i][1]
        s = stitch(s, projs[i+1], (oy, ox))
    # print(s)
    cv2.imwrite("test_stitch.png", s)
