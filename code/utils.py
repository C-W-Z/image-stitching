import os
import cv2
import numpy as np
from feature import DescriptorType, MotionType
from stitch import BlendingType

def perror(message:str):
    print(message)
    exit()

def to_bool(value:str, file:str, line:int):
    if value.capitalize() == 'True' or value == '1':
        return True
    elif value.capitalize() == 'False' or value == '0':
        return False
    else:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid bool value")

def to_int(value:str, file:str, line:int):
    try:
        return int(value)
    except ValueError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid int value")

def to_float(value:str, file:str, line:int):
    try:
        return float(value)
    except ValueError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid float value")

def to_DescriptorType(value:str, file:str, line:int):
    try:
        return DescriptorType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid DescriptorType value")

def to_MotionType(value:str, file:str, line:int):
    try:
        return MotionType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid MotionType value")

def to_BlendingType(value:str, file:str, line:int):
    try:
        return BlendingType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid BlendingType value")

def read_images(image_list:str) -> tuple[list[np.ndarray[np.uint8,3]], list[float]]:

    images = []
    focals = []
    is360 = False
    scales = 1
    scale_sigma = 1
    harris_sigma = 0.5
    thres_ratio = 0.1
    grid_size = 20
    descriptor = DescriptorType.SIFT
    feature_match_thres = 0.8
    motion = MotionType.TRANSLATION
    ransac_thres = 1
    ransac_iter = 1000
    blend = BlendingType.LINEAR
    crop = True

    input_dir = os.path.dirname(image_list)

    try:
        with open(image_list, 'r') as img_list:
            for i, line in enumerate(img_list):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue

                if line.startswith('IS360'):
                    line.replace(' ', '')
                    is360 = to_bool(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('SCALES'):
                    line.replace(' ', '')
                    scales = to_int(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('SCALE_SIGMA'):
                    line.replace(' ', '')
                    scale_sigma = to_float(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('HARRIS_SIGMA'):
                    line.replace(' ', '')
                    harris_sigma = to_float(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('THRES_RATIO'):
                    line.replace(' ', '')
                    thres_ratio = to_float(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('GRID_SIZE'):
                    line.replace(' ', '')
                    grid_size = to_int(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('DESCIPTOR'):
                    line.replace(' ', '')
                    descriptor = to_DescriptorType(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('FEATURE_MATCH_THRES'):
                    line.replace(' ', '')
                    feature_match_thres = to_float(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('MOTION'):
                    line.replace(' ', '')
                    motion = to_MotionType(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('RANSAC_THRES'):
                    line.replace(' ', '')
                    ransac_thres = to_float(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('RANSAC_ITER'):
                    line.replace(' ', '')
                    ransac_iter = to_int(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('BLEND'):
                    line.replace(' ', '')
                    blend = to_BlendingType(line.split('=')[1].strip(), image_list, i)

                elif line.startswith('CROP'):
                    line.replace(' ', '')
                    crop = to_bool(line.split('=')[1].strip(), image_list, i)

                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename, f, *_ = parts
                        filepath = os.path.join(input_dir, filename)
                        print(f"reading file {filepath}")
                        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                        if img is None:
                            perror(f"Error: Can not read file {filepath}")
                        images.append(img)
                        focals.append(to_float(f, image_list, i))
                    else:
                        perror(f"Error in {image_list}, line {i+1}: Not enough arguments")

        assert(len(images) == len(focals))
        return images, focals, scales, is360, scale_sigma, harris_sigma, thres_ratio, grid_size, descriptor, feature_match_thres, motion, ransac_thres, ransac_iter, blend, crop

    except FileNotFoundError as e:
        perror(f"FileNotFoundError: {e}")

def crop_vertical(image:np.ndarray[np.uint8,3]):
    non_zero_rows = np.where(image[:, :, 3].sum(axis=1) > 0)[0]
    top, bottom = non_zero_rows[0], non_zero_rows[-1]
    return image[top:bottom+1, :]

def crop_horizontal(image:np.ndarray[np.uint8,3]):
    non_zero_cols = np.where(image[:, :, 3].sum(axis=0) > 0)[0]
    left, right = non_zero_cols[0], non_zero_cols[-1]
    return image[:, left:right+1]

def crop_transparency(image:np.ndarray[np.uint8,3]):
    non_zero_rows = np.where(image[:, :, 3].sum(axis=1) > 0)[0]
    non_zero_cols = np.where(image[:, :, 3].sum(axis=0) > 0)[0]
    top, bottom = non_zero_rows[0], non_zero_rows[-1]
    left, right = non_zero_cols[0], non_zero_cols[-1]
    return image[top:bottom+1, left:right+1]

def crop_rectangle(image:np.ndarray[np.uint8,3]):
    alpha_channel = image[:,:,3]
    H, *_ = alpha_channel.shape
    alpha_row = np.sum(alpha_channel != 0, axis=1)
    max_row = np.max(alpha_row)
    top_row = np.argmax(alpha_row == max_row)
    bottom_row = np.argmax(alpha_row[::-1] == max_row)

    return image[top_row:H-bottom_row]

def cylindrical_projection(image:np.ndarray[np.uint8,3], focal:float) -> np.ndarray[np.uint8, 3]:
    H, W, *_ = image.shape
    proj = np.zeros((H, W, 4), dtype=np.uint8) # add alpha channel
    y_coords, x_coords = np.ogrid[:H, :W]
    _y = y_coords - H // 2
    _x = x_coords - W // 2
    theta = np.arctan(_x / focal)
    h = _y / np.sqrt(_x ** 2 + focal ** 2)
    X = W // 2 + (focal * theta).astype(int)
    Y = H // 2 + (focal * h).astype(int)
    # X = np.clip(X, 0, W - 1)
    # Y = np.clip(Y, 0, H - 1)
    proj[Y, X, :3] = image[y_coords, x_coords, :]
    proj[Y, X, 3] = 255 # alpha = 255
    return crop_horizontal(proj)

def rotate_image(image:np.ndarray[np.uint8,3], angle:float, centerXY:tuple[float,float]=None):
    H, W, *_ = image.shape
    if centerXY == None:
        centerXY = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(centerXY, angle, 1)
    return cv2.warpAffine(image, M, (W, H))

def normalize(image:np.ndarray[np.uint8,3]):
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # return (image - image.mean()) / image.std()

def draw_keypoints(image:np.ndarray[np.uint8,3], keypoints:list[tuple[int,int]], angles:list[float] | None, filename:str=None, arrow_length=10):
    assert(angles is None or len(keypoints) == len(angles))
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(len(keypoints)):
        y, x = keypoints[i]
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
        if angles is None:
            continue
        end_x = int(x + arrow_length * np.cos(angles[i] * np.pi / 180))
        end_y = int(y - arrow_length * np.sin(angles[i] * np.pi / 180))
        cv2.arrowedLine(image, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 1)

    if not filename is None:
        cv2.imwrite(f"{filename}.jpg", image)

    return image

def draw_matches(img_left:np.ndarray[np.uint8,3], point_left:list[tuple[int,int]], img_right:np.ndarray[np.uint8,3], point_right:list[tuple[int,int]], inliers:list[int]=[], outliers:list[int]=[], filename:str=None):
    HL, WL, CL = img_left.shape
    HR, WR, CR = img_right.shape
    assert(CL == CR and CR == 4)
    N = len(point_left)
    assert(N == len(point_right))
    image = np.zeros((max(HL,HR), WL+WR, 4), dtype=np.uint8)
    image[:HL, :WL] = img_left
    image[:HR, WL:WL+WR] = img_right

    def draw_colors(indices:list[int], color:tuple[int,int,int]):
        for i in indices:
            yl, xl = point_left[i]
            yl, xl = int(yl), int(xl)
            yr, xr = point_right[i]
            yr, xr = int(yr), WL + int(xr)
            cv2.line(image, (xl, yl), (xr, yr), color, 1)

    if len(inliers) == 0 or len(outliers) == 0:
        draw_colors(range(N), (0, 0, 255))
    else:
        draw_colors(outliers, (0, 0, 255))
        draw_colors(inliers, (0, 255, 0))

    if not filename is None:
        cv2.imwrite(f"{filename}.jpg", image)

    return image

def gaussian_weights(shape, centerYX, sigma):
    """
    Generate Gaussian weights centered at centerYX = (y, x) with given sigma.
    """
    x_indices = np.arange(shape[1])
    y_indices = np.arange(shape[0])

    x_indices, y_indices = np.meshgrid(x_indices, y_indices)

    return np.exp(-((x_indices - centerYX[0])**2 + (y_indices - centerYX[1])**2) / (2 * sigma**2))

def to_multi_scale(gray:np.ndarray[np.uint8,2], nums:int=2, sigma:float=1):
    grays = [gray]
    ksize = (5, 5)
    for _ in range(1, nums):
        gray = cv2.GaussianBlur(gray, ksize, sigmaX=sigma, sigmaY=sigma)
        H, W = gray.shape
        gray = cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        grays.append(gray)
    return grays

def check_and_make_dir(dir:str):
    if os.path.isdir(dir):
        return
    print(f"{dir} is not a directory")
    try:
        print(f"Making directory {dir} ...")
        os.makedirs(dir, exist_ok=True)
        print(f"Success")
    except Exception as e:
        perror(f"Error: {e}")

if __name__ == '__main__':
    imgs, focals, *_ = read_images("data\parrington\list.txt")
    proj = cylindrical_projection(imgs[0], focals[0])
    cv2.imwrite("test.jpg", proj)
