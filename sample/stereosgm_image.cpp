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

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libsgm.h>

#include "sample_common.h"

static const std::string keys =
"{ @left_img   | <none> | path to input left image                                                            }"
"{ @right_img  | <none> | path to input right image                                                           }"
"{ disp_size   |     64 | maximum possible disparity value                                                    }"
"{ P1          |     10 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
"{ P2          |    120 | penalty on the disparity change by more than 1 between neighbor pixels              }"
"{ uniqueness  |   0.95 | margin in ratio by which the best cost function value should be at least second one }"
"{ num_paths   |      8 | number of scanlines used in cost aggregation                                        }"
"{ min_disp    |      0 | minimum disparity value                                                             }"
"{ LR_max_diff |      1 | maximum allowed difference between left and right disparity                         }"
"{ census_type |      1 | type of census transform (0:CENSUS_9x7 1:SYMMETRIC_CENSUS_9x7)                      }"
"{ help h      |        | display this help and exit                                                          }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	cv::Mat I1 = cv::imread(parser.get<cv::String>("@left_img"), cv::IMREAD_UNCHANGED);
	cv::Mat I2 = cv::imread(parser.get<cv::String>("@right_img"), cv::IMREAD_UNCHANGED);

	const int disp_size = parser.get<int>("disp_size");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const float uniqueness = parser.get<float>("uniqueness");
	const int num_paths = parser.get<int>("num_paths");
	const int min_disp = parser.get<int>("min_disp");
	const int LR_max_diff = parser.get<int>("LR_max_diff");
	const auto census_type = static_cast<sgm::CensusType>(parser.get<int>("census_type"));

	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");
	ASSERT_MSG(census_type == sgm::CensusType::CENSUS_9x7 || census_type == sgm::CensusType::SYMMETRIC_CENSUS_9x7, "census type must be 0 or 1.");

	const int src_depth = I1.type() == CV_8U ? 8 : 16;
	const int dst_depth = 16;
	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;

	const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, false, path_type, min_disp, LR_max_diff, census_type);
	sgm::StereoSGM ssgm(I1.cols, I1.rows, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);

	cv::Mat disparity(I1.size(), CV_16S);

	ssgm.execute(I1.data, I2.data, disparity.data);

	// create mask for invalid disp
	const cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	// show image
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_TURBO);
	disparity_8u.setTo(0, mask);
	disparity_color.setTo(cv::Scalar::all(0), mask);
	if (I1.type() != CV_8U)
		cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX, CV_8U);

	const std::vector<cv::Mat> images = { disparity_8u, disparity_color, I1 };
	const std::vector<std::string> titles = { "disparity", "disparity color", "input" };

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
