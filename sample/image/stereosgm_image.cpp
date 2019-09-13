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

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv,
		"{@left_img   | <none> | path to input left image                                                            }"
		"{@right_img  | <none> | path to input right image                                                           }"
		"{disp_size   |     64 | maximum possible disparity value                                                    }"
		"{P1          |     10 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
		"{P2          |    120 | penalty on the disparity change by more than 1 between neighbor pixels              }"
		"{uniqueness  |   0.95 | margin in ratio by which the best cost function value should be at least second one }"
		"{num_paths   |      8 | number of scanlines used in cost aggregation                                        }"
		"{min_disp    |      0 | minimum disparity value                                                             }"
		"{LR_max_diff |      1 | maximum allowed difference between left and right disparity                         }"
		"{help h      |        | display this help and exit                                                          }");

	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	cv::Mat  left = cv::imread(parser.get<cv::String>("@left_img"), -1);
	cv::Mat right = cv::imread(parser.get<cv::String>("@right_img"), -1);

	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	const int disp_size = parser.get<int>("disp_size");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const float uniqueness = parser.get<float>("uniqueness");
	const int num_paths = parser.get<int>("num_paths");
	const int min_disp = parser.get<int>("min_disp");
	const int LR_max_diff = parser.get<int>("LR_max_diff");

	ASSERT_MSG(!left.empty() && !right.empty(), "imread failed.");
	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");

	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
	const int input_depth = left.type() == CV_8U ? 8 : 16;
	const int output_depth = 16;

	const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, false, path_type, min_disp, LR_max_diff);
	sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);

	cv::Mat disparity(left.size(), CV_16S);

	ssgm.execute(left.data, right.data, disparity.data);

	// create mask for invalid disp
	cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	// show image
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	disparity_8u.setTo(0, mask);
	disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
	if (left.type() != CV_8U)
		cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8U);

	std::vector<cv::Mat> images = { disparity_8u, disparity_color, left };
	std::vector<std::string> titles = { "disparity", "disparity color", "input" };

	std::cout << "Hot keys:" << std::endl;
	std::cout << "\tESC - quit the program" << std::endl;
	std::cout << "\ts - switch display (disparity | colored disparity | input image)" << std::endl;

	int mode = 0;
	while (true) {

		cv::setWindowTitle("image", titles[mode]);
		cv::imshow("image", images[mode]);

		const char c = cv::waitKey(0);
		if (c == 's')
			mode = (mode < 2 ? mode + 1 : 0);
		if (c == 27)
			break;
	}

	return 0;
}
