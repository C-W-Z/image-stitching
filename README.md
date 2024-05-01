# Image Stitching

CSIE B11902078 張宸瑋

CSIE B10902058 胡桓碩

Assignment description: [https://www.csie.ntu.edu.tw/~cyy/courses/vfx/24spring/assignments/proj2/](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/24spring/assignments/proj2/)

## Dependencies

- opencv-python
- numpy
- scipy

## Usage

```shell
$ cd code
$ python main.py -h
usage: main.py [-h] [-d] <input_file> <output_directory>

Read images & arguments from information in <input_file> & output the panoramic image 'panorama_*parameters*.png'   
to <output_directory>

positional arguments:
  <input_file>        Input file (.txt) path
  <output_directory>  Output directory path

options:
  -h, --help          show this help message and exit
  -d                  Output debug images in <output_directory>
```

For example:

```shell
$ cd code
$ python main.py ../data/grail/origin/list.txt ../data/grail
```

## Input File

For example, this is the file [data/grail/origin/list.txt](data/grail/list.txt)

```txt
# whether this photo set cover 360 degree
IS360 = True # (default False)

# how many scales used (for multi-scale)
SCALES = 2 # (default 1)

# sigma for Guassian blur when multi-scaling the photos
SCALE_SIGMA = 1 # (default 1.0)

# sigma for Guassian blur in harris dectector
HARRIS_SIGMA = 0.5 # (default 0.5)

# threshold ratio used in harris dectector
# threshold = THRES_RATIO * max corner response
THRES_RATIO = 0.1 # (default 0.1)

# grid size in non-maximum suppression
GRID_SIZE = 20 # (default 20)

# use SIFT or MSOP feature descriptor
DESCIPTOR = MSOP # (default SIFT)

# threshold used in feature matching 
# distance to the closest feature should < FEATURE_MATCH_THRES * distance to the second closest feature
FEATURE_MATCH_THRES = 0.8 # (default 0.8)

# TRANSLATION / AFFINE / PERSPECTIVE motion
MOTION = TRANSLATION # (default TRANSLATION)

# the threshold used in RANSAC
RANSAC_THRES = 2 # (default 1.0)

# the iteration number of RANSAC
RANSAC_ITER = 5000 # (default 1000)

# SEAM (seam finding) or LINEAR (linear blending) or NONE
BLEND = SEAM # (default LINEAR)

# auto adjust the brightness between multiple images
AUTO_EXPOSURE = True # (default False)

# crop panorama to a rectangle
CROP = True # (default True)

# image     focal
grail17.jpg 629.796
grail16.jpg 630.357
grail15.jpg 630.007
grail14.jpg 630.933
grail13.jpg 628.722
grail12.jpg 629.814
grail11.jpg 628.981
grail10.jpg 629.02
grail09.jpg 628.635
grail08.jpg 627.979
grail07.jpg 627.7
grail06.jpg 625.495
grail05.jpg 627.858
grail04.jpg 627.719
grail03.jpg 627.378
grail02.jpg 626.08
grail01.jpg 626.377
grail00.jpg 628.919
```

Note that the `#` is just like a comment in python, we will ignore everything after `#`.

The input file should be a .txt file, for example `list.txt`.

There are some parameters:

|Parameter|Option|Default|Description
|---|---|---|---|
IS360|`True` `False`|`False`|whether this photo set cover 360 degree
SCALES|int, >=1|1|how many scales used for multi-scale
SCALE_SIGMA|float, >=0|1.0|sigma for Guassian blur when multi-scaling the photos
HARRIS_SIGMA|float, >=0|0.5|sigma for Guassian blur in harris dectector
THRES_RATIO|float, 0~1|0.1|threshold ratio used in harris dectector, `threshold = THRES_RATIO * max corner response`
GRID_SIZE|int, >=1|20|grid size in non-maximum suppression
DESCIPTOR|`MSOP` `SIFT`|`MSOP`|use SIFT or MSOP feature descriptor
FEATURE_MATCH_THRES|float 0~1|0.8|threshold used in feature matching, `distance to the closest feature should < FEATURE_MATCH_THRES * distance to the second closest feature`
MOTION|`TRANSLATION` `AFFINE` `PERSPECTIVE`|`TRANSLATION`|the motion model used for matching and stitching photos
RANSAC_THRES|float, >=0|1.0|the threshold used in RANSAC, determine a motion is inlier or outlier
RANSAC_ITER|int, >=1|the iteration number of RANSAC
BLEND|`NONE` `SEAM` `LINEAR`|`LINEAR`|no blending, seam finding, or linear blending when stitching photos|
AUTO_EXPOSURE|`True` `False`|`False`|auto adjust the brightness between multiple images or not
CROP|`True` `False`|`False`|crop panorama to a rectangle or not

Last, we need to write the image file names and the focal length as follow.

```txt
# filenames      focal lengths
your-image-1.jpg XXX
your-image-2.jpg XXX
your-image-3.jpg XXX
```

Note that the order of the filenames should be clockwise, which means the first image should be the left most image and the last image should be the right most one. And the overlapping region of each two images should preferably not exceed half.


The file names should not contain any spaces, and this .txt file must be in the same folder as the images you write in it.

The structure of folder will be like:

```txt
origin/
  list.txt
  your-image-1.jpg
  your-image-2.jpg
  your-image-3.jpg
```
