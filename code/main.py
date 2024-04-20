import utils, stitch
from feature import *
import harris

if __name__ == '__main__':
    imgs, focals = utils.read_images("data\parrington\list.txt")

    imgs = imgs[0:2]
    focals = focals[0:2]
    N = len(imgs)
    H, W, _ = imgs[0].shape
    S = 2
    # 360
    full_rotate = False

    imgs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]
    # imgs = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2BGRA) for i in range(N)]
    print("Complete Cylindrical Projection")

    grays = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in imgs]
    multi_grays = [utils.to_multi_scale(g, S, 1) for g in grays]

    multi_keypoints = [harris.multi_scale_harris(g, 0.5, 0.1, 20) for g in multi_grays]
    print("Complete Harris Detection")

    # Descriptor
    multi_point = []
    multi_desc = []
    multi_orien = []
    for i in range(N):
        multi_g = multi_grays[i]
        multi_p = multi_keypoints[i]
        assert(S == len(multi_g) and S == len(multi_p))
        point = []
        desc = []
        orien = []
        for s in range(S):
            multi_p[s] = subpixel_refinement(multi_g[s], multi_p[s])
            p, d, o = sift_descriptor(multi_g[s], multi_p[s])
            point.append(p)
            desc.append(d)
            orien.append(o)
            print(f"Complete {len(p)} feature descriptors in image {i}, scale {s}")
            utils.draw_keypoints(multi_g[s], p, o, f"test{i}_{s}")
        multi_point.append(point)
        multi_desc.append(desc)
        multi_orien.append(orien)

    # Matching
    # Ms = []
    offsets = []
    for i in range(N):
        if not full_rotate and i == N - 1:
            break
        desc1 = multi_desc[i]
        desc2 = multi_desc[(i + 1) % N]
        point1 = multi_point[i]
        point2 = multi_point[(i + 1) % N]
        orien1 = multi_orien[i]
        orien2 = multi_orien[(i + 1) % N]
        points1 = []
        points2 = []
        oriens1 = []
        oriens2 = []
        for s in range(S):
            matches = feature_matching(desc1[s], desc2[s], 0.85)
            idx1, idx2 = np.asarray(matches).T
            m_point1 = point1[s][idx1] * (1 << s)
            m_point2 = point2[s][idx2] * (1 << s)
            m_orien1 = orien1[s][idx1]
            m_orien2 = orien2[s][idx2]
            m_point1 = subpixel_refinement(grays[i], m_point1)
            m_point2 = subpixel_refinement(grays[(i + 1) % N], m_point2)
            points1.extend(m_point1)
            points2.extend(m_point2)
            oriens1.extend(m_orien1)
            oriens2.extend(m_orien2)

        utils.draw_keypoints(imgs[i], points1, oriens1, f"testmatch{i}_left")
        utils.draw_keypoints(imgs[(i + 1) % N], points2, oriens2, f"testmatch{i+1}_right")

        # left image - right image
        # the keypoints are at the right part of left image and left part of right image
        sample_offsets = np.array(points1) - np.array(points2)
        offset = stitch.ransac_translation(sample_offsets, 1, 5000)
        offsets.append(offset)

        # M = stitch.ransac_homography(matched_keypoints2, matched_keypoints1, 1, 5000)
        # M, _ = cv2.findHomography(matched_keypoints2[:, ::-1], matched_keypoints1[:, ::-1], cv2.RANSAC, ransacReprojThreshold=1, confidence=0.999)
        # M, _ = cv2.estimateAffinePartial2D(matched_keypoints2[:, ::-1], matched_keypoints1[:, ::-1], method=cv2.RANSAC, ransacReprojThreshold=1 ,confidence=0.999)
        # Ms.append(M)

    print("offsets =", offsets)
    s = stitch.stitch_all_horizontal(imgs, offsets, stitch.BlendingType.SeamFinding, full_rotate)
    # s = stitch.stitch_all_homography(imgs, Ms)
    cv2.imwrite("test.png", s)
