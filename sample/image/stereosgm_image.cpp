/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION == 2
#include <opencv2/contrib/contrib.hpp>
#endif

#include <libsgm.h>

cv::Mat color_normalized(const cv::Mat& src)
{
	cv::Mat tmp(src.size(), CV_8UC1);
	double min, max;
	cv::minMaxIdx(src, &min, &max);
	src.convertTo(tmp, CV_8UC1, 256 / max, -min);
	cv::Mat dst;
	cv::applyColorMap(tmp, dst, cv::COLORMAP_JET);
	return dst;
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img right_img [disp_size]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cv::Mat left = cv::imread(argv[1], -1);
	cv::Mat right = cv::imread(argv[2], -1);

	int disp_size = 64;
	if (argc >= 4) {
		disp_size = atoi(argv[3]);
	}
	const int out_depth = argc > 4 ? std::stoi(argv[4]) : 8;
	const bool subpixel = argc > 5 ? std::stoi(argv[5]) != 0 : false;

	if (left.size() != right.size() || left.type() != right.type()) {
		std::cerr << "mismatch input image size" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int bits = 0;

	switch (left.type()) {
	case CV_8UC1: bits = 8; break;
	case CV_16UC1: bits = 16; break;
	default:
		std::cerr << "invalid input image color format" << left.type() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const sgm::StereoSGM::Parameters params{10, 120, 0.95f, subpixel};

	sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, bits, out_depth, sgm::EXECUTE_INOUT_HOST2HOST, params);

	cv::Mat output(cv::Size(left.cols, left.rows), out_depth == 8 ? CV_8UC1 : CV_16UC1);

	ssgm.execute(left.data, right.data, output.data);
	// show image
	cv::Mat dst;
	if (subpixel) {
		dst = output * (1 << out_depth) / (disp_size * sgm::StereoSGM::SUBPIXEL_SCALE);
	} else {
		dst = output * (1 << out_depth) / disp_size;
	}
	cv::imshow("image", dst);

	cv::Mat colored = color_normalized(output);
	int key = cv::waitKey();
	int mode = 0;
	while (key != 27) {
		if (key == 's') {
			mode += 1;
			if (mode >= 3) mode = 0;

			switch (mode) {
			case 0:
				{
					#if CV_MAJOR_VERSION == 3
					cv::setWindowTitle("image", "disparity");
					#endif
					cv::imshow("image", dst);
					break;
				}
			case 1:
				{
					#if CV_MAJOR_VERSION == 3
					cv::setWindowTitle("image", "disparity color");
					#endif
					cv::imshow("image", colored);
					break;
				}
			case 2:
				{
					#if CV_MAJOR_VERSION == 3
					cv::setWindowTitle("image", "input");
					#endif
					cv::imshow("image", left);
					break;
				}
			}
		}
		key = cv::waitKey();
	}
}
