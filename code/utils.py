import os
import cv2

def perror(message:str):
    print(message)
    exit()

def read_images(image_list):

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
                    focals.append(f)
                else:
                    perror(f"Error in {image_list}, line {i+1}: Not enough arguments")

        assert(len(images) == len(focals))
        return (images, focals)

    except FileNotFoundError as e:
        perror(f"FileNotFoundError: {e}")

if __name__ == '__main__':
    read_images("data\parrington\list.txt")
