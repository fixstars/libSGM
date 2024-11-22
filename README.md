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
- sgm path : 4 path
- subpixel : enabled

### Results
|Device|CUDA version|Processing Time[Milliseconds]|FPS|
|---|---|---|---|
|GTX 1080 Ti|10.1|2.0|495.1|
|GeForce RTX 3080|11.1|1.5|651.3|
|Tegra X2|10.0|28.5|35.1|
|Xavier(MODE_15W)|10.2|17.3|57.7|
|Xavier(MAXN)|10.2|9.0|110.7|

## Requirements
|Package Name|Minimum Requirements|Note
|---|---|---|
|CMake|version >= 3.18||
|CUDA Toolkit|compute capability >= 3.5|
|OpenCV|version >= 3.4.8|for samples|
|OpenCV CUDA module|version >= 3.4.8|for OpenCV wrapper|
|ZED SDK|version >= 3.0|for ZED sample|
|Pybind11|latest|for python bindings|

## Build Instructions
```
$ git clone https://github.com/fixstars/libSGM.git
$ cd libSGM
$ git submodule update --init  # It is needed if ENABLE_TESTS option is set to ON
$ mkdir build
$ cd build
$ cmake ../  # Several options available
$ make
```

## Sample Execution
```
$ pwd
.../libSGM
$ cd build
$ cmake .. -DENABLE_SAMPLES=on
$ make
$ cd sample
$ ./stereosgm_movie <left image path format> <right image path format> <disparity_size>
left image path format: the format used for the file paths to the left input images
right image path format: the format used for the file paths to the right input images
disparity_size: the maximum number of disparities (optional)
```

"disparity_size" is optional. By default, it is 128.

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
$ ./stereosgm_movie left_image_%04d.pgm right_image_%04d.pgm
```

The sample images available at [Daimler Urban Scene Segmentation Benchmark Dataset 2014](http://www.6d-vision.com/scene-labeling) are used to test the software.

## Test Execution
libSGM uses [Google Test](https://github.com/google/googletest) for tests as Git submodule.  
So, we need to init submodule by following command firstly.

```
$ pwd
.../libSGM
$ git submodule update --init
```

We can run tests after a build.

```
$ pwd
.../libSGM
$ cd build
$ cd test
$ ./sgm-test
```

Test code compares our implementation of each functions to naive implementation.

## Author
The "adaskit Team"  

The adaskit is an open-source project created by [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

## License
Apache License 2.0
