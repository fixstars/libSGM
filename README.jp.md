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
- sgm path : 4 path
- subpixel : enabled

### Results
|Device|Processing Time[Milliseconds]|FPS|
|---|---|---|
|Tegra X2 (CUDA: v10.0)|28.5|35.1|
|GTX 1080 Ti (CUDA: v10.1)|2.0|495.1|

## Requirements
libSGMはCUDA (compute capabilities >= 3.5)を必要とします。  
また、サンプルをビルドする際には以下のライブラリが必要となります。
- OpenCV 3.0 以降
- CMake 3.1 以降

## build
```
$ git clone https://github.com/fixstars/libSGM.git
$ cd libSGM
$ git submodule update --init  # ENABLE_TESTS オプションを ON にする際に必要です
$ mkdir build
$ cd build
$ cmake ../  # いくつかのオプションが用意されています
$ make
```

## サンプル実行
```
$ pwd
.../libSGM
$ cd build
$ cd sample/movie/
$ ./stereo_movie <left image path format> <right image path format> <disparity_size>
left image path format: 左側画像入力時に使用するファイルパスのフォーマット
right image path format: 右側画像入力時に使用するファイルパスのフォーマット
disparity_size: 視差情報を何段階で保持するか(省略可)
```

disparity_sizeは省略が可能です。省略した時には、128が付与されます。

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

本ソフトウェアは [Daimler Urban Scene Segmentation Benchmark Dataset 2014](http://www.6d-vision.com/scene-labeling) にて提供されている画像を用いて動作確認をしています。

## Test Execution
libSGMでは[Google Test](https://github.com/google/googletest)をテスト・フレームワークとして採用しています。  
Git submodule機能を通して導入しているため、始めに以下のコマンドで初期化する必要があります。

```
$ pwd
.../libSGM
$ git submodule update --init
```

ビルド後、以下のコマンドでテストを実行できます。

```
$ pwd
.../libSGM
$ cd build
$ cd test
$ ./sgm-test
```

テストコードではナイーブな実装との比較を行っています。

## Authors
The "SGM Team": Samuel Audet, Yoriyuki Kitta, Yuta Noto, Ryo Sakamoto, Akihiro Takagi  
[Fixstars Corporation](http://www.fixstars.com/)

## License
Apache License 2.0
