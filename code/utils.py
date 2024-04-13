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

def cylindrical_projection(image:np.ndarray[np.uint8,3], focal:float) -> np.ndarray[np.uint8, 3]:
    H, W, _ = image.shape
    proj = np.zeros(image.shape, dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            _y = y - int(H / 2)
            _x = x - int(W / 2)
            theta = np.arctan(_x / focal)
            h = _y / np.sqrt(_x ** 2 + focal ** 2)
            X = int(W / 2) + int(focal * theta)
            Y = int(H / 2) + int(focal * h)
            proj[Y, X, :] = image[y, x, :]
    return proj

if __name__ == '__main__':
    imgs, focals = read_images("data\parrington\list.txt")
    proj = cylindrical_projection(imgs[0], focals[0])
    cv2.imwrite("test.jpg", proj)
