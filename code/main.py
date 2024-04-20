import utils, stitch
from feature_matching import *
import harris

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")

    imgs = imgs[0:2]
    focals = focals[0:2]
    N = len(imgs)
    H, W, _ = imgs[0].shape

    projs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]
    # projs = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2BGRA) for i in range(N)]
    print("Complete Cylindrical Projection")
    grays = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in projs]
    keypoints = [harris.harris_detector(gray, 0.5, 0.001, 15) for gray in grays]
    print("Complete Harris Detection")

    descs = []
    points = []
    orientations = []
    for i in range(N):
        keypoints[i] = subpixel_refinement(grays[i], keypoints[i])
        # utils.draw_keypoints(projs[i], keypoints[i], None, f"test{i}")
        p, d, o = sift_descriptor(grays[i], keypoints[i])
        points.append(p)
        descs.append(d)
        orientations.append(o)
        print("Complete Feature Description:", len(p))
        utils.draw_keypoints(projs[i], p, o, f"test{i}")

    # Ms = []
    offsets = []
    for i in range(N-1):
        matches = feature_matching(descs[i], descs[(i + 1) % N], 0.85)
        match_idx1 = np.array([i for i, _ in matches], dtype=np.int32)
        match_idx2 = np.array([j for _, j in matches], dtype=np.int32)
        matched_keypoints1 = points[i][match_idx1]
        orientations1 = orientations[i][match_idx1]
        matched_keypoints2 = points[(i + 1) % N][match_idx2]
        orientations2 = orientations[(i + 1) % N][match_idx2]
        utils.draw_keypoints(projs[0], matched_keypoints1, orientations1, "testmatch0.jpg")
        utils.draw_keypoints(projs[1], matched_keypoints2, orientations2, "testmatch1.jpg")

        # left image - right image
        # the keypoints are at the right part of left image and left part of right image
        sample_offsets = matched_keypoints1 - matched_keypoints2
        offset = stitch.ransac_translation(sample_offsets, 1, 5000)
        offsets.append(offset)

        # M = stitch.ransac_homography(matched_keypoints2, matched_keypoints1, 1, 5000)
        # M, _ = cv2.findHomography(matched_keypoints2[:, ::-1], matched_keypoints1[:, ::-1], cv2.RANSAC, ransacReprojThreshold=1, confidence=0.999)
        # M, _ = cv2.estimateAffinePartial2D(matched_keypoints2[:, ::-1], matched_keypoints1[:, ::-1], method=cv2.RANSAC, ransacReprojThreshold=1 ,confidence=0.999)
        # Ms.append(M)

    print("offsets =", offsets)
    s = stitch.stitch_horizontal(projs[0], projs[1], offsets[0], stitch.BlendingType.SeamFinding)
    # s = stitch.stitch_all_horizontal(projs, offsets, True)
    # s = stitch.stitch_all_homography(projs, Ms)
    cv2.imwrite("test_1_2.png", s)
