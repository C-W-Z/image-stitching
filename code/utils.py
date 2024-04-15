import os
import cv2
import numpy as np

def perror(message:str):
    print(message)
    exit()

def to_float(value:str, file:str, line:int):
    try:
        return float(value)
    except ValueError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid float value")

def read_images(image_list:str) -> tuple[list[np.ndarray[np.uint8,3]], list[float]]:

    images = []
    focals = []

    input_dir = os.path.dirname(image_list)

    try:
        with open(image_list, 'r') as img_list:
            for i, line in enumerate(img_list):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue

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
        return (images, focals)

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

def draw_keypoints(image:np.ndarray[np.uint8,3], keypoints:list[tuple[int,int]], angles:list[float], filename:str=None):
    assert(angles is None or len(keypoints) == len(angles))
    image = image.copy()

    arrow_length = 10
    for i in range(len(keypoints)):
        y, x = keypoints[i]
        cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
        if angles is None:
            continue
        end_x = int(x + arrow_length * np.cos(angles[i] * np.pi / 180))
        end_y = int(y - arrow_length * np.sin(angles[i] * np.pi / 180))
        cv2.arrowedLine(image, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 1)

    cv2.imwrite(f"{filename}.jpg", image)

def gaussian_weights(shape, centerYX, sigma):
    """
    Generate Gaussian weights centered at centerYX = (y, x) with given sigma.
    """
    x_indices = np.arange(shape[1])
    y_indices = np.arange(shape[0])

    x_indices, y_indices = np.meshgrid(x_indices, y_indices)

    return np.exp(-((x_indices - centerYX[0])**2 + (y_indices - centerYX[1])**2) / (2 * sigma**2))

if __name__ == '__main__':
    imgs, focals = read_images("data\parrington\list.txt")
    proj = cylindrical_projection(imgs[0], focals[0])
    cv2.imwrite("test.jpg", proj)
