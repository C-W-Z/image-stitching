import argparse
import os
import cv2
import numpy as np
import utils, harris, feature, stitch
from feature import DescriptorType, MotionType
from stitch import BlendingType

def main(input_file:str, output_dir:str, debug:bool=False):
    imgs, focals, S, IS360, scale_sigma, harris_sigma, thres_ratio, grid_size, descriptor, feature_match_thres, MOTION, ransac_thres, ransac_iter, BLEND, AUTO_EXP, CROP = utils.read_images(input_file)

    # imgs = imgs[0:2]
    # focals = focals[0:2]
    N = len(imgs)
    # H, W, _ = imgs[0].shape

    imgs = [utils.cylindrical_projection(imgs[i], focals[i]) for i in range(N)]
    # imgs = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2BGRA) for i in range(N)]
    print("Complete Cylindrical Projection")

    grays = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in imgs]
    multi_grays = [utils.to_multi_scale(g, S, scale_sigma) for g in grays]

    multi_keypoints = [harris.multi_scale_harris(g, harris_sigma, thres_ratio, grid_size) for g in multi_grays]
    if debug:
        for i in range(N):
            for s in range(S):
                utils.draw_keypoints(multi_grays[i][s], multi_keypoints[i][s], None, os.path.join(output_dir, f"harris_{i}_{s}"))
    print("Complete Harris Detection")

    # Descriptor
    multi_point = []
    multi_desc = []
    # multi_orien = []
    for i in range(N):
        multi_g = multi_grays[i]
        multi_p = multi_keypoints[i]
        assert(S == len(multi_g) and S == len(multi_p))
        point = []
        desc = []
        # orien = []
        for s in range(S):
            multi_p[s] = feature.subpixel_refinement(multi_g[s], multi_p[s])
            if descriptor == DescriptorType.SIFT:
                p, d, o = feature.sift_descriptor(multi_g[s], multi_p[s])
            elif descriptor == DescriptorType.MSOP:
                p, d, o = feature.msop_descriptor(multi_g[s], multi_p[s])
            point.append(p)
            desc.append(d)
            # orien.append(o)
            print(f"Complete {len(p)} feature descriptors in image {i}, scale {s}")
            # if debug:
            #     utils.draw_keypoints(multi_g[s], p, o, os.path.join(output_dir, f"keypoints_{i}_{s}"))
        multi_point.append(point)
        multi_desc.append(desc)
        # multi_orien.append(orien)

    # Matching
    Ms = []
    offsets = []
    for i in range(N):
        if i == N - 1 and (not IS360 or MOTION != MotionType.TRANSLATION):
            break
        desc1 = multi_desc[i]
        desc2 = multi_desc[(i + 1) % N]
        point1 = multi_point[i]
        point2 = multi_point[(i + 1) % N]
        # orien1 = multi_orien[i]
        # orien2 = multi_orien[(i + 1) % N]
        points1 = []
        points2 = []
        # oriens1 = []
        # oriens2 = []
        for s in range(S):
            matches = feature.feature_matching(desc1[s], desc2[s], feature_match_thres)
            idx1, idx2 = np.asarray(matches).T
            m_point1 = point1[s][idx1] * (1 << s)
            m_point2 = point2[s][idx2] * (1 << s)
            # m_orien1 = orien1[s][idx1]
            # m_orien2 = orien2[s][idx2]
            m_point1 = feature.subpixel_refinement(grays[i], m_point1)
            m_point2 = feature.subpixel_refinement(grays[(i + 1) % N], m_point2)
            points1.extend(m_point1)
            points2.extend(m_point2)
            # oriens1.extend(m_orien1)
            # oriens2.extend(m_orien2)

        points1 = np.array(points1)
        points2 = np.array(points2)
        inliers = []
        outliers = []

        if MOTION == MotionType.TRANSLATION:
            # left image - right image
            # the keypoints are at the right part of left image and left part of right image
            sample_offsets = points1 - points2
            offset, inliers, outliers = stitch.ransac_translation(sample_offsets, ransac_thres, ransac_iter)
            offsets.append(offset)
        elif MOTION == MotionType.AFFINE:
            # M = stitch.ransac_affine(points2, points1, ransac_thres, ransac_iter)
            if i < N // 2:
                M, _ = cv2.estimateAffinePartial2D(points1[:, ::-1], points2[:, ::-1], method=cv2.RANSAC, ransacReprojThreshold=ransac_thres, confidence=0.999)
            else:
                M, _ = cv2.estimateAffinePartial2D(points2[:, ::-1], points1[:, ::-1], method=cv2.RANSAC, ransacReprojThreshold=ransac_thres, confidence=0.999)
            if debug:
                print(M)
            Ms.append(M)
        elif MOTION == MotionType.PERSPECTIVE:
            if i < N // 2:
                M, inliers, outliers = stitch.ransac_homography(points1, points2, ransac_thres, ransac_iter)
                # M, _ = cv2.findHomography(points1[:, ::-1], points2[:, ::-1], cv2.RANSAC, ransacReprojThreshold=ransac_thres, confidence=0.9999)
            else:
                M, inliers, outliers = stitch.ransac_homography(points2, points1, ransac_thres, ransac_iter)
                # M, _ = cv2.findHomography(points2[:, ::-1], points1[:, ::-1], cv2.RANSAC, ransacReprojThreshold=ransac_thres, confidence=0.9999)
            if debug:
                print(M)
            Ms.append(M)

        if debug:
            utils.draw_matches(imgs[i], points1, imgs[(i + 1) % N], points2, inliers, outliers, os.path.join(output_dir, f"match_keypoints_{i}_{(i + 1) % N}"))

    if MOTION == MotionType.TRANSLATION:
        if debug:
            offset = np.asarray(offsets)
            print("offsets =", offset.tolist())
            if BLEND == BlendingType.SEAM:
                tmp = [np.copy(img) for img in imgs]
                s = stitch.stitch_all_horizontal(tmp, np.copy(offsets), BLEND, IS360, AUTO_EXP, True)
                cv2.imwrite(os.path.join(output_dir, "seam.png"), s)

        s = stitch.stitch_all_horizontal(imgs, offsets, BLEND, IS360, AUTO_EXP)

        if CROP:
            s = utils.crop_rectangle(s)
    else:
        s = stitch.stitch_all_homography(imgs, Ms)

    print("Complete Image Stitching")

    if MOTION == MotionType.TRANSLATION:
        filename = f"panoramic_{IS360}_{S}_{scale_sigma}_{harris_sigma}_{thres_ratio}_{grid_size}_{descriptor}_{feature_match_thres}_{MOTION}_{ransac_thres}_{ransac_iter}_{BLEND}_{CROP}.png"
    elif MOTION == MotionType.AFFINE:
        filename = f"panoramic_{S}_{scale_sigma}_{harris_sigma}_{thres_ratio}_{grid_size}_{descriptor}_{feature_match_thres}_{MOTION}_{ransac_thres}.png"
    elif MOTION == MotionType.PERSPECTIVE:
        filename = f"panoramic_{S}_{scale_sigma}_{harris_sigma}_{thres_ratio}_{grid_size}_{descriptor}_{feature_match_thres}_{MOTION}_{ransac_thres}_{ransac_iter}.png"

    filename = os.path.join(output_dir, filename)
    cv2.imwrite(filename, s)
    print(f"Save panorama into {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read images & arguments from information in <input_file> & output the panoramic image 'panorama_*parameters*.png' to <output_directory>\n")
    parser.add_argument("input_file", type=str, metavar="<input_file>", help="Input file (.txt) path")
    parser.add_argument("output_directory", type=str, metavar="<output_directory>", help="Output directory path")
    parser.add_argument("-d", action="store_true", help="Output feature points and other debug images in <output_directory>")
    args = parser.parse_args()
    utils.check_and_make_dir(args.output_directory)
    main(args.input_file, args.output_directory, args.d)
