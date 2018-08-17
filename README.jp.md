# libSGM
---
Semi-Global MatchingをベースとしたCUDA実装です。

## 概要
---

libSGMは、Semi-Global MatchingアルゴリズムをCUDAで実装したものです。  
適切にキャリブレーションされた2つの入力画像から、視差画像を取得することができます。

## 特徴
---
CUDAを使用し、高速な視差画像算出が可能

## パフォーマンス
benchmarkサンプルで計測した処理時間を示します
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
libSGMはCUDA (compute capabilities >= 3.0)を必要とします。  
また、サンプルをビルドする際には以下のライブラリが必要となります。
- OpenCV
- OpenGL
- GLFW3
- GLEW
- CMake 3.10 以降

## build
```
$ git clone https://github.com/fixstars/libSGM.git
$ cd libSGM
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## サンプル実行
```
$ pwd
.../libSGM
$ cd build
$ cd sample/movie/
$ ./stereo_movie <left image path format> <right image path format> <disparity> <frame count>
left image path format: 左側画像入力時に使用するファイルパスのフォーマット
right image path format: 右側画像入力時に使用するファイルパスのフォーマット
disparity: 視差情報を何段階で保持するか(省略可)
frame count: 全画像が何枚存在するか(省略可)
```

disparityとframe countは省略が可能です。省略した時には、それぞれ64, 100が付与されます。

ここで、left image path format, right image path formatとは、ファイル読み込み時に使用するフォーマットを意味します。  
次のような連番ファイルが与えられていたとき、与えるべきpath formatは以下のようになります。
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

movie, imageサンプルは、
http://www.6d-vision.com/scene-labeling
にて提供されている、Daimler Urban Scene Segmentation Benchmark Datasetにて
動作確認をしています。

## Authors
The "SGM Team": Samuel Audet, Yoriyuki Kitta, Yuta Noto, Ryo Sakamoto, Akihiro Takagi  
[Fixstars Corporation](http://www.fixstars.com/)

## License
Apache License 2.0
