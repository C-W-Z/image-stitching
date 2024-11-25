import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.spatial.distance import euclidean
import utils
from feature import *
from enum import IntEnum

class BlendingType(IntEnum):
    NONE = 0
    LINEAR = 1
    SEAM = 2
    def __str__(self):
        return self.name.upper()

def ransac_translation(
    offsets:np.ndarray[float,2],
    threshold:float,
    iterations:int=1000
):
    N = len(offsets)
    best_offset = None
    max_inlier_count = -1
    for i in range(min(N, iterations)):
        if iterations < N:
            i = np.random.randint(0, N)
        inlier_count = 0
        for j in range(N):
            if i == j:
                continue
            if euclidean(offsets[i], offsets[j]) <= threshold:
                inlier_count += 1
        if max_inlier_count < inlier_count:
            max_inlier_count = inlier_count
            best_offset = offsets[i]

    print(f"Find {max_inlier_count} inliers in {N} matches")

    # calculate average offset
    totalY = 0
    totalX = 0
    count = 0
    inliers = []
    outliers = []
    for i in range(N):
        if euclidean(best_offset, offsets[i]) <= threshold:
            totalY += offsets[i][0]
            totalX += offsets[i][1]
            count += 1
            inliers.append(i)
        else:
            outliers.append(i)

    return np.array([totalY / count, totalX / count]), inliers, outliers

def seam_finding(
    img_left:np.ndarray[np.uint8,3],
    img_right:np.ndarray[np.uint8,3],
    show_seam:bool=False
):
    assert(img_left.shape == img_right.shape)
    overlap_w = img_left.shape[1]
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2GRAY)
    diff_map = np.abs(gray_left - gray_right)
    H, W = diff_map.shape
    assert(W == overlap_w)
    # map[y, x] = diff_map[y, x] + diff[y, x+1]
    new_map = diff_map[:, :-1] + diff_map[:, 1:]
    H, W = new_map.shape
    assert(W == overlap_w - 1)

    # Dynamic Programming
    cumulative_map = np.zeros_like(new_map, dtype=np.float32)
    cumulative_map[0] = new_map[0]
    for i in range(1, H):
        for j in range(W):
            # Choose the minimum cumulative energy from previous row
            prev_cumulative_diff = cumulative_map[i-1, max(j-1, 0):min(j+2, W)]
            cumulative_map[i, j] = new_map[i, j] + np.min(prev_cumulative_diff)

    seam = []
    min_index = np.argmin(cumulative_map[-1])
    seam.append(min_index)
    for i in range(H-2, -1, -1):
        j = seam[-1]
        prev_cumulative_diff = cumulative_map[i, max(j-1, 0):min(j+2, W)]
        min_index = np.argmin(prev_cumulative_diff) + max(j-1, 0)
        seam.append(min_index)
    seam = np.array(seam)
    # seam = np.flip(seam)
    result = np.zeros_like(img_left)
    H, W, C = img_left.shape
    for y in range(H-1, -1, -1):
        for c in range(C):
            result[y, :, c] = np.where(np.arange(W) <= seam[y], img_left[y, :, c], img_right[y, :, c])
        if show_seam:
            result[y, seam[y]] = [0, 0, 255, 255]

    return result

def stitch_horizontal(
    img_left:np.ndarray[np.uint8,3],
    img_right:np.ndarray[np.uint8,3],
    offset:np.ndarray[float,2],
    blending:BlendingType,
    auto_exposure:bool=False,
    show_seam:bool=False
):
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
        warp_right = cv2.warpAffine(img_right, M, (new_W, new_H))

        # copy left to combined image
        warp_left = np.zeros((new_H, new_W, 4), dtype=np.float32)
        warp_left[:HL, :WL] = img_left

    else:
        new_H = max(HL, HR) + int(np.ceil(abs(dy)))

        # shift the right image
        M = np.float32([[1, 0, dx],
                        [0, 1,  0]])
        warp_right = cv2.warpAffine(img_right, M, (new_W, new_H))

        # shift the left image and output to combined image
        M = np.float32([[1, 0,  0],
                        [0, 1, -dy]])
        warp_left = cv2.warpAffine(img_left, M, (new_W, new_H))

    # clear the translucent coords caused by warpAffine (since dy, dx are float)
    warp_right[warp_right[:, :, 3] < 225] = 0
    warp_left[warp_left[:, :, 3] < 225] = 0

    # the overlap x indices are dx ~ WL
    overlap_x = int(np.floor(dx))

    left_alpha = warp_left[:, overlap_x:WL, 3] > 127 # img_left coords with alpha > 127
    right_alpha = warp_right[:, overlap_x:WL, 3] > 127 # img_right coords with  alpha > 127

    # find the true overlap coords
    true_overlap = np.where(np.logical_and(left_alpha, right_alpha))
    translated_true_overlap = (true_overlap[0], overlap_x + true_overlap[1])

    # auto exposure
    if auto_exposure:
        warp_left = warp_left.astype(np.float32)
        warp_right = warp_right.astype(np.float32)
        for c in range(3):
            gray_left = warp_left[:,:,c] + 1e-6
            gray_right = warp_right[:,:,c] + 1e-6
            left_mean = np.mean(gray_left[translated_true_overlap])
            right_mean = np.mean(gray_right[translated_true_overlap])
            mean_exposure = (left_mean + right_mean) / 2
            left_diff = left_mean - mean_exposure
            right_diff = right_mean - mean_exposure
            warp_left[:,:,c] -= left_diff
            warp_right[:,:,c] -= right_diff

    # copy the right no overlap area
    combined = np.copy(warp_left)
    combined[:, WL:] = warp_right[:, WL:]

    if blending == BlendingType.LINEAR:
        overlap_weights = np.linspace(0, 1, WL - overlap_x)
        # expand (repeat) overlap_weights to shape (newH, len(overlap_weights), 4)
        overlap_weights = np.tile(overlap_weights.reshape(1, -1, 1), (new_H, 1, 4))

        # linear blending
        combined[translated_true_overlap] = (
            warp_left[translated_true_overlap] * (1 - overlap_weights[true_overlap]) + 
            warp_right[translated_true_overlap] * overlap_weights[true_overlap]
        )

    elif blending == BlendingType.SEAM:
        combined[:, overlap_x:WL] = seam_finding(warp_left[:, overlap_x:WL], warp_right[:, overlap_x:WL], show_seam)

    else:
        combined[translated_true_overlap] = warp_right[translated_true_overlap]

    # the coords that inside overlap area but only the img_right has value (alpha > 127)
    right_nolap = np.where(np.logical_and(np.logical_not(left_alpha), right_alpha))
    right_nolap = (right_nolap[0], overlap_x + right_nolap[1])
    combined[right_nolap] = warp_right[right_nolap]

    # the coords that inside overlap area but only the img_left has value (alpha > 127)
    left_nolap = np.where(np.logical_and(left_alpha, np.logical_not(right_alpha)))
    left_nolap = (left_nolap[0], overlap_x + left_nolap[1])
    combined[left_nolap] = warp_left[left_nolap]

    return combined

def end_to_end_align(
    panorama:np.ndarray[np.uint8,3],
    offsetY:float
):
    # print(offsetY)
    H, W, C = panorama.shape
    dy = np.linspace(0, offsetY, W, dtype=np.float32)
    oy = int(np.ceil(np.abs(offsetY)))
    align = np.zeros((H + oy, W, C), dtype=np.float32)
    if offsetY > 0:
        align[oy:oy+H] = panorama
    else:
        align[:H] = panorama
    for x in range(W):
        align[:,x] = shift(align[:,x], (-dy[x], 0), mode='nearest', order=1)
    return align.astype(np.uint8)

def stitch_all_horizontal(
    images:np.ndarray[np.uint8,3],
    offsets:np.ndarray[float,2],
    blending:BlendingType,
    end_to_end:bool=False,
    auto_exposure:bool=False,
    show_seam:bool=False
):
    N = len(images)

    if N == len(offsets):
        s = stitch_horizontal(images[-1], images[0], offsets[-1], blending, auto_exposure, show_seam)
        divideX = s.shape[1] - images[-1].shape[1]
        if offsets[-1][0] > 0:
            offsets[0][0] += offsets[-1][0]
        images[0] = s[:, divideX:]
        images[-1] = s[:, :divideX]

    s = images[0]
    oy, ox = 0, 0
    for i, offset in enumerate(offsets):
        if i == N - 1:
            break
        oy += offset[0]
        ox += offset[1]
        s = stitch_horizontal(s, images[i + 1], (oy, ox), blending, auto_exposure, show_seam)

    if auto_exposure:
        s[:,:,:3] = utils.normalize(s[:,:,:3]) * 255

    if end_to_end:
        s = end_to_end_align(s, oy - offsets[-1][0])

    s = utils.crop_vertical(s)
    return s.astype(np.uint8)

def ransac_affine(
    srcpoints:np.ndarray[float,2],
    dstpoints:np.ndarray[float,2],
    threshold:float,
    iterations:int=1000
):
    assert(srcpoints.shape == dstpoints.shape)
    N = len(srcpoints)
    assert(N >= 3)
    srcpoints = srcpoints[:, ::-1] # [(y,x)] to [(x,y)]
    dstpoints = dstpoints[:, ::-1] # [(y,x)] to [(x,y)]
    best_H = None
    max_inlier_count = 0
    best_inliers = []
    for _ in range(iterations):
        indices = np.random.choice(N, 3, replace=False)
        inlier_count = 0
        # H = dlt(srcpoints[indices], dstpoints[indices])
        H, _ = cv2.estimateAffinePartial2D(srcpoints[indices], dstpoints[indices])
        if H is None or np.nan in H or np.inf in H:
            continue
        # inliers = indices.copy()
        inliers = np.empty((0,), dtype=np.int32)
        for j in range(N):
            # if j in indices:
            #     continue
            src = np.append(srcpoints[j], 1)
            dst = (H @ src.T).T[:2]
            if euclidean(dstpoints[j], dst) <= threshold:
                inlier_count += 1
                inliers = np.append(inliers, j)
        if max_inlier_count < inlier_count:
            max_inlier_count = inlier_count
            best_H = H
            best_inliers = inliers

    print(f"Find {max_inlier_count} inliers in {N} matches")
    assert(max_inlier_count >= 3)
    # print(best_inliers)
    # best_H = dlt(srcpoints[best_inliers], dstpoints[best_inliers])
    H, _ = cv2.estimateAffinePartial2D(srcpoints[best_inliers], dstpoints[best_inliers])
    if not (H is None or np.nan in H or np.inf in H):
        best_H = H
    return best_H

def ransac_homography(
    srcpoints:np.ndarray[float,2],
    dstpoints:np.ndarray[float,2],
    threshold:float,
    iterations:int=1000
):
    assert(srcpoints.shape == dstpoints.shape)
    N = len(srcpoints)
    assert(N >= 4)
    srcpoints = srcpoints[:, ::-1] # [(y,x)] to [(x,y)]
    dstpoints = dstpoints[:, ::-1] # [(y,x)] to [(x,y)]
    best_H = None
    max_inlier_count = 0
    best_inliers = []
    for _ in range(iterations):
        indices = np.random.choice(N, 4, replace=False)
        inlier_count = 0
        # H = dlt(srcpoints[indices], dstpoints[indices])
        H, _ = cv2.findHomography(srcpoints[indices], dstpoints[indices])
        if H is None or np.nan in H or np.inf in H:
            continue
        # inliers = indices.copy()
        inliers = np.empty((0,), dtype=np.int32)
        for j in range(N):
            # if j in indices:
            #     continue
            src = np.append(srcpoints[j], 1)
            dst = (H @ src.T).T[:2]
            if euclidean(dstpoints[j], dst) <= threshold:
                inlier_count += 1
                inliers = np.append(inliers, j)
        if max_inlier_count < inlier_count:
            max_inlier_count = inlier_count
            best_H = H
            best_inliers = inliers

    best_outliers = list(set(range(N)) - set(best_inliers))

    print(f"Find {max_inlier_count} inliers in {N} matches")
    assert(max_inlier_count >= 4)
    # print(best_inliers)
    # best_H = dlt(srcpoints[best_inliers], dstpoints[best_inliers])
    H, _ = cv2.findHomography(srcpoints[best_inliers], dstpoints[best_inliers])
    if not (H is None or np.nan in H or np.inf in H):
        best_H = H
    return best_H, best_inliers, best_outliers

def estimate_transformed_corners(
    H:int,
    W:int,
    M:np.ndarray[float,2]
):
    corners = np.array([[0, 0, 1], [0, H-1, 1], [W-1, 0, 1], [W-1, H-1, 1]], dtype=np.float32)
    transformed_corners = (M @ corners.T)[:2].T
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])
    return min_x, max_x, min_y, max_y

def stitch_homography(
    src:np.ndarray[np.uint8,3],
    dst:np.ndarray[np.uint8,3],
    M:np.ndarray[float,2]
):
    HS, WS, CS = src.shape
    HD, WD, CD = dst.shape
    assert(CS == CD and CS == 4)

    if M.shape == (2, 3): # Affine
        M = np.append(M, np.array([[0, 0, 1]]), axis=0)

    min_x, max_x, min_y, max_y = estimate_transformed_corners(HS, WS, M)

    origin_W = max(WS, WD)
    new_W = origin_W
    left = 0
    if min_x < 0:
        left = int(np.ceil(np.abs(min_x)))
        new_W += left
    if max_x > origin_W:
        new_W += int(np.ceil(max_x)) - origin_W

    origin_H = max(HS, HD)
    new_H = origin_H
    top = 0
    if min_y < 0:
        top = int(np.ceil(np.abs(min_y)))
        new_H += top
    if max_y > origin_H:
        new_H += int(np.ceil(max_y)) - origin_H

    result = np.zeros((new_H, new_W, 4), dtype=np.uint8)
    result[top:top+HS, left:left+WS] = src
    result = cv2.warpPerspective(result, M, (new_W, new_H))
    # clear the translucent coords caused by warpPerspective
    result[result[:, :, 3] < 225] = 0

    alpha = np.where(dst[:, :, 3] > 127)
    translated_alpha = (alpha[0] + top, alpha[1] + left)
    result[translated_alpha] = dst[alpha]
    return result

def stitch_all_homography(
    images:np.ndarray[np.uint8,3],
    Ms:list[np.ndarray[float,3]]
):
    N = len(images)
    mid = N // 2

    # s = images[0]
    s = images[mid]
    H = np.eye(3)
    for i in range(mid, N-1):
        M = Ms[i]
        if M.shape == (2, 3): # Affine
            M = np.append(M, np.array([[0, 0, 1]]), axis=0)
        H = H @ M
        s = stitch_homography(images[i + 1], s, H)

    H = np.eye(3)
    for i in range(mid-1, -1, -1):
        M = Ms[i]
        if M.shape == (2, 3): # Affine
            M = np.append(M, np.array([[0, 0, 1]]), axis=0)
        H = M
        s = stitch_homography(images[i], s, H)

    # for i, M in enumerate(Ms):
    #     if i == N - 1:
    #         break
    #     if M.shape == (2, 3): # Affine
    #         M = np.append(M, np.array([[0, 0, 1]]), axis=0)
    #     H = H @ M
    #     s = stitch_homography(images[i + 1], s, H)

    s = utils.crop_transparency(s)
    return s
