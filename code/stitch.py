import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.spatial.distance import euclidean
import utils
from feature_matching import *
from Harris_by_ShuoEn import *
from dlt import dlt

def ransac_translation(offsets:np.ndarray[float,2], threshold:float, iterations:int=1000):
    N = len(offsets)
    best_offset = None
    max_inliner_count = -1
    for _ in range(iterations):
        i = np.random.randint(0, N)
        inliner_count = 0
        for j in range(N):
            if i == j:
                continue
            if euclidean(offsets[i], offsets[j]) <= threshold:
                inliner_count += 1
        if max_inliner_count < inliner_count:
            max_inliner_count = inliner_count
            best_offset = offsets[i]

    print(f"Find {max_inliner_count} inliners in {N} matches")

    # calculate average offset
    totalY = 0
    totalX = 0
    count = 0
    for i in range(N):
        if euclidean(best_offset, offsets[i]) <= threshold:
            totalY += offsets[i][0]
            totalX += offsets[i][1]
            count += 1

    return (totalY / count, totalX / count)

def stitch_horizontal(img_left:np.ndarray[np.uint8,3], img_right:np.ndarray[np.uint8,3], offset:tuple[float,float]):
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

        # shift the right image
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        warped_right = cv2.warpAffine(img_right, M, (new_W, new_H))

        # copy left to combined image
        combined = np.zeros((new_H, new_W, 4), dtype=np.uint8)
        combined[:HL, :WL] = img_left

    else:
        new_H = max(HL, HR) + int(np.ceil(abs(dy)))

        # shift the right image
        M = np.float32([[1, 0, dx],
                        [0, 1,  0]])
        warped_right = cv2.warpAffine(img_right, M, (new_W, new_H))

        # shift the left image and output to combined image
        M = np.float32([[1, 0,  0],
                        [0, 1, -dy]])
        combined = cv2.warpAffine(img_left, M, (new_W, new_H))

    # copy the right no overlap area
    combined[:, WL:] = warped_right[:, WL:]

    # clear the translucent coords caused by warpAffine (since dy, dx are float)
    warped_right[warped_right[:, :, 3] < 225] = 0
    combined[combined[:, :, 3] < 225] = 0

    # the overlap x indices are dx ~ WL
    overlap_x = int(np.floor(dx))

    left_alpha = combined[:, overlap_x:WL, 3] > 127 # img_left coords with alpha > 127
    right_alpha = warped_right[:, overlap_x:WL, 3] > 127 # img_right coords with  alpha > 127

    # the coords that inside overlap area but only the img_right has value (alpha > 127)
    right_nolap = np.where(np.logical_and(np.logical_not(left_alpha), right_alpha))
    right_nolap = (right_nolap[0], overlap_x + right_nolap[1])
    combined[right_nolap] = warped_right[right_nolap]

    # find the true overlap coords
    true_overlap = np.where(np.logical_and(left_alpha, right_alpha))
    translated_true_overlap = (true_overlap[0], overlap_x + true_overlap[1])

    overlap_weights = np.linspace(0, 1, WL - overlap_x)
    # expand (repeat) overlap_weights to shape (newH, len(overlap_weights), 4)
    overlap_weights = np.tile(overlap_weights.reshape(1, -1, 1), (new_H, 1, 4))

    # linear blending
    combined[translated_true_overlap] = (
        combined[translated_true_overlap] * (1 - overlap_weights[true_overlap]) + 
        warped_right[translated_true_overlap] * overlap_weights[true_overlap]
    )

    combined[combined[:, :, 3] < 226] = 0
    combined[combined[:, :, 3] > 225, 3] = 255
    return combined

def end_to_end_align(panorama:np.ndarray[np.uint8,3], offsetY:float):
    # print(offsetY)
    H, W, C = panorama.shape
    dy = np.linspace(0, offsetY, W, dtype=np.float32)
    oy = int(np.ceil(offsetY))
    align = np.zeros((H + oy, W, C), dtype=np.float32)
    align[oy:oy+H] = panorama
    for x in range(W):
        align[:,x] = shift(align[:,x], (-dy[x], 0), mode='nearest', order=1)
    return align.astype(np.uint8)

def stitch_all(images:np.ndarray[np.uint8,3], offsets:np.ndarray[float,2], end_to_end:bool=False):
    N = len(offsets)
    assert(N == len(images))

    if end_to_end:
        s = stitch_horizontal(images[-1], images[0], offsets[-1])
        images[-1] = s[:, :s.shape[0]//2]
        images[0] = s[:, s.shape[0]//2:]

    s = images[0]
    oy, ox = 0, 0
    for i, offset in enumerate(offsets):
        if i == N - 1:
            break
        oy += offset[0]
        ox += offset[1]
        s = stitch_horizontal(s, images[i + 1], (oy, ox))

    if end_to_end:
        s = end_to_end_align(s, oy - offsets[-1][0])

    s = crop_vertical(s)
    print("Complete Image Stitching")
    return s

def ransac_homography(srcpoints:np.ndarray[float,2], dstpoints:np.ndarray[float,2], threshold:float, iterations:int=1000):
    assert(srcpoints.shape == dstpoints.shape)
    N = len(srcpoints)
    srcpoints = srcpoints[:, ::-1] # [(y,x)] to [(x,y)]
    dstpoints = dstpoints[:, ::-1] # [(y,x)] to [(x,y)]
    # best_H = None
    max_inliner_count = 0
    best_inliners = None
    for _ in range(iterations):
        indices = np.random.choice(N, 4, replace=False)
        inliner_count = 0
        H = dlt(srcpoints[indices], dstpoints[indices])
        if np.nan in H or np.inf in H:
            continue
        inliners = indices.copy()
        for j in range(N):
            if j in indices:
                continue
            src = np.append(srcpoints[j], 1)
            dst = (H @ src.T).T[:2]
            if euclidean(dstpoints[j], dst) <= threshold:
                inliner_count += 1
                np.append(inliners, j)
        if max_inliner_count < inliner_count:
            max_inliner_count = inliner_count
            # best_H = H
            best_inliners = inliners

    print(f"Find {max_inliner_count} inliners in {N} matches")
    H = dlt(srcpoints[best_inliners], dstpoints[best_inliners])
    return H

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")

    N = len(imgs)
    H, W, _ = imgs[0].shape

    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]

    # offsets = [(5.1, 251.0), (3.740740740740741, 242.0), (4.9523809523809526, 245.28571428571428), (4.125, 249.03125), (3.888888888888889, 240.03703703703704), (4.911764705882353, 246.1764705882353), (3.8333333333333335, 247.79166666666666), (5.04, 239.12), (4.0476190476190474, 244.04761904761904), (4.84, 246.88), (4.25, 241.0), (3.076923076923077, 249.92307692307693), (4.916666666666667, 240.95833333333334), (2.8461538461538463, 247.69230769230768), (5.6875, 240.125), (5.833333333333333, 242.94444444444446), (3.857142857142857, 245.0)]

    # offsets = [(5.0, 246.3), (4.230769230769231, 239.92307692307693), (3.142857142857143, 250.28571428571428), (3.8, 244.0), (4.428571428571429, 249.85714285714286), (3.75, 241.25), (3.3333333333333335, 243.33333333333334), (4.0, 244.5), (3.3636363636363638, 238.0909090909091), (4.125, 252.875), (4.0, 242.83333333333334), (3.857142857142857, 245.21428571428572), (5.181818181818182, 249.0), (2.8823529411764706, 239.94117647058823), (5.0, 245.92857142857142), (3.4166666666666665, 248.08333333333334), (5.142857142857143, 238.57142857142858), (3.9375, 244.875)]

    offsets = [(4.7631578947368425, 246.92105263157896), (4.056603773584905, 241.1320754716981), (3.9545454545454546, 249.9090909090909), (4.276595744680851, 241.06382978723406), (4.173076923076923, 248.90384615384616), (4.14, 241.04), 
(4.05, 244.01666666666668), (4.176470588235294, 244.94117647058823), (4.0, 239.16216216216216), (4.057142857142857, 
252.05714285714285), (4.1875, 242.125), (4.183333333333334, 245.06666666666666), (4.160714285714286, 249.25), (4.078125, 240.046875), (4.69811320754717, 245.0943396226415), (4.111111111111111, 248.88888888888889), (4.104166666666667, 239.125), (3.8666666666666667, 244.35555555555555)]

    s = stitch_all(projs, offsets, True)
    cv2.imwrite("test_stitch.png", s)
