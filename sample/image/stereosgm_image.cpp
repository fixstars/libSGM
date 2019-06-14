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

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

int main(int argc, char* argv[]) {
	cv::CommandLineParser parser(argc, argv,
		"{@left_img  | <none> | path to input left image                                                            }"
		"{@right_img | <none> | path to input right image                                                           }"
		"{disp_size  |     64 | maximum possible disparity value                                                    }"
		"{P1         |     10 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
		"{P2         |    120 | penalty on the disparity change by more than 1 between neighbor pixels              }"
		"{uniqueness |   0.95 | margin in ratio by which the best cost function value should be at least second one }"
		"{num_paths  |      8 | number of scanline used in optimization of cost function                            }"
		"{help h     |        | display this help and exit                                                          }");
	
	if (parser.has("help")) {
		parser.printMessage();
		return EXIT_SUCCESS;
	}

	const cv::Mat  left = cv::imread(parser.get<cv::String>( "@left_img"), -1);
	const cv::Mat right = cv::imread(parser.get<cv::String>("@right_img"), -1);

	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		return EXIT_FAILURE;
	}

	const int disp_size = parser.get<int>("disp_size");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const float uniqueness = parser.get<float>("uniqueness");
	const int num_paths = parser.get<int>("num_paths");

	ASSERT_MSG(!left.empty() && !right.empty(), "imread failed.");
	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scan path must be 4 or 8.");

	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;

	int bits = 0;

	switch (left.type()) {
	case CV_8UC1: bits = 8; break;
	case CV_16UC1: bits = 16; break;
	default:
		std::cerr << "invalid input image color format" << left.type() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	sgm::StereoSGM ssgm(
		left.cols, left.rows, disp_size,
		bits, 8, sgm::EXECUTE_INOUT_HOST2HOST,
		sgm::StereoSGM::Parameters(P1, P2, uniqueness, false, path_type));

	cv::Mat output(cv::Size(left.cols, left.rows), CV_8UC1);

	ssgm.execute(left.data, right.data, output.data);
	// show image
	cv::imshow("image", output * 256 / disp_size);
	
	int key = cv::waitKey();
	int mode = 0;
	while (key != 27) {
		if (key == 's') {
			mode += 1;
			if (mode >= 3) mode = 0;

			switch (mode) {
			case 0:
				{
					cv::setWindowTitle("image", "disparity");
					cv::imshow("image", output * 256 / disp_size);
					break;
				}
			case 1:
				{
					cv::Mat m;
					cv::applyColorMap(output * 256 / disp_size, m, cv::COLORMAP_JET);
					cv::setWindowTitle("image", "disparity color");
					cv::imshow("image", m);
					break;
				}
			case 2:
				{
					cv::setWindowTitle("image", "input");
					cv::imshow("image", left);
					break;
				}
			}
		}
		key = cv::waitKey();
	}

	return 0;
}
