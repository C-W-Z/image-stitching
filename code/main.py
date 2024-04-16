import utils, stitch
from feature_matching import *
from Harris_by_ShuoEn import *

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")

    N = len(imgs)
    H, W, _ = imgs[0].shape

    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]
    print("Complete Cylindrical Projection")
    keypoints = [harris_detector(img, 0.1) for img in projs]
    print("Complete Harris Detection")

    descs = []
    points = []
    orientations = []
    for i in range(N):
        gray = cv2.cvtColor(projs[i], cv2.COLOR_BGRA2GRAY)
        keypoints[i] = subpixel_refinement(gray, keypoints[i])
        # utils.draw_keypoints(projs[i], keypoints[i], None, f"test{i}")
        p, d, o = sift_descriptor(gray, keypoints[i])
        points.append(p)
        descs.append(d)
        orientations.append(o)
        print("Complete Feature Description:", len(p))
        utils.draw_keypoints(projs[i], p, o, f"test{i}")

    offsets = []
    for i in range(N):
        matches = feature_matching(descs[i], descs[(i + 1) % N], 0.8)
        match_idx1 = np.array([i for i, _ in matches], dtype=np.int32)
        match_idx2 = np.array([j for _, j in matches], dtype=np.int32)
        matched_keypoints1 = points[i][match_idx1]
        # orientations1 = orientations[i][match_idx1]
        matched_keypoints2 = points[(i + 1) % N][match_idx2]
        # orientations2 = orientations[(i + 1) % N][match_idx2]

        # left image - right image
        # the keypoints are at the right part of left image and left part of right image
        sample_offsets = matched_keypoints1 - matched_keypoints2
        offset = stitch.ransac_translation(sample_offsets, 0.5, 1000)
        offsets.append(offset)

    print("offsets =", offsets)

    s = stitch.stitch_all(projs, offsets, True)
    cv2.imwrite("test_parrington.png", s)
