# libSGM
---
A CUDA implementation performing Semi-Global Matching.

## Introduction
---

libSGM is library that implements in CUDA the Semi-Global Matching algorithm.  
From a pair of appropriately calibrated input images, we can obtain the disparity map.

## Features
---
Because it uses CUDA, we can compute the disparity map at high speed.

## Performance
The libSGM performance obtained from benchmark sample
### Settings
- image size : 1024 x 440
- disparity size : 128
- sgm path : 8 path

### Results
|Device|Processing Time[Milliseconds]|FPS|
|---|---|---|
|Tegra X2|52.4|19.1|
|GTX 1080 Ti|3.4|296|

## Requirements
libSGM needs CUDA (compute capabilities >= 3.0) to be installed.  
Moreover, to build the sample, we need the following libraries:
- OpenCV 3.0 or later
- CMake 3.10 or later

## Build Instructions
```
$ git clone https://github.com/fixstars/libSGM.git
$ cd libSGM
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## Sample Execution
```
$ pwd
.../libSGM
$ cd build
$ cd sample/movie/
$ ./stereo_movie <left image path format> <right image path format> <disparity> <frame count>
left image path format: the format used for the file paths to the left input images
right image path format: the format used for the file paths to the right input images
disparity: the maximum number of disparities (optional)
frame count: the total number of images (optional)
```

"disparity" and "frame count" are optional. By default, they are 64 and 100, respectively.

Next, we explain the meaning of the "left image path format" and "right image path format".  
When provided with the following set of files, we should pass the "path formats" given below.
```
left_image_0000.pgm
left_image_0001.pgm
left_image_0002.pgm
left_image_0003.pgm
...

right_image_0000.pgm
right_image_0001.pgm
right_image_0002.pgm
right_image_0003.pgm
```

```
$ ./stereo_movie left_image_%04d.pgm right_image_%04d.pgm
```

The sample movie images available at
http://www.6d-vision.com/scene-labeling
under "Daimler Urban Scene Segmentation Benchmark Dataset"
are used to test the software.

## Authors
The "SGM Team": Samuel Audet, Yoriyuki Kitta, Yuta Noto, Ryo Sakamoto, Akihiro Takagi  
[Fixstars Corporation](http://www.fixstars.com/)

## License
Apache License 2.0
