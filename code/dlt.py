import numpy as np
import numpy.linalg as la

def normalize_points(points:np.ndarray[float,2]) -> tuple[np.ndarray[float,2],np.ndarray[float],float]:
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    avg_distance = np.mean(la.norm(centered_points, axis=1))
    scale_factor = np.sqrt(2) / avg_distance

    affine = np.array([[scale_factor, 0, -scale_factor * mean_point[0]],
                       [0, scale_factor, -scale_factor * mean_point[1]],
                       [0,            0,                             1]])

    # [[x,y]  to [[x x x]
    #  [x,y]      [y y y]
    #  [x,y]]     [1 1 1]]
    ones = np.ones((1, points.shape[0]))
    points = np.append(points.T, ones, axis=0)

    points = affine @ points
    points = points[:2].T

    return points, affine

def denormalize_H(H:np.ndarray[float,2], affine_src:np.ndarray[float,2], affine_dst:np.ndarray[float,2]) -> np.ndarray[float,2]:
    return la.inv(affine_dst) @ H @ affine_src

def dlt(srcpointsXY:np.ndarray[float,2], dstpointsXY:np.ndarray[float,2]):
    assert(srcpointsXY.shape == dstpointsXY.shape)

    normalized_src, affine_src = normalize_points(srcpointsXY)
    normalized_dst, affine_dst = normalize_points(dstpointsXY)

    A = []
    for i in range(4):
        x, y = normalized_src[i]
        u, v = normalized_dst[i]
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    A = np.array(A)

    _, _, V = la.svd(A)
    H = V[-1, :].reshape(3, 3)
    H /= H[-1, -1] # the last element in 3x3 matrix should be 1

    return denormalize_H(H, affine_src, affine_dst)
