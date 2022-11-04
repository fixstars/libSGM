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
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libsgm.h>

#include "sample_common.h"

static const std::string keys =
"{@left-image-format  | <none> | format string for path to input left image  }"
"{@right-image-format | <none> | format string for path to input right image }"
"{disp_size           |    128 | maximum possible disparity value            }"
"{start_number        |      0 | index to start reading                      }"
"{help h              |        | display this help and exit                  }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	if (argc < 3 || parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	const std::string image_format_L = parser.get<cv::String>("@left-image-format");
	const std::string image_format_R = parser.get<cv::String>("@right-image-format");
	const int disp_size = parser.get<int>("disp_size");
	const int start_number = parser.get<int>("start_number");

	cv::Mat I1 = cv::imread(cv::format(image_format_L.c_str(), start_number), cv::IMREAD_UNCHANGED);
	cv::Mat I2 = cv::imread(cv::format(image_format_R.c_str(), start_number), cv::IMREAD_UNCHANGED);

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

	const int width = I1.cols;
	const int height = I1.rows;

	const int src_depth = I1.type() == CV_8U ? 8 : 16;
	const int dst_depth = disp_size < 256 ? 8 : 16;
	const int src_bytes = src_depth * width * height / 8;
	const int dst_bytes = dst_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
	cv::Mat disparity(height, width, dst_depth == 8 ? CV_8S : CV_16S), disparity_color;

	const int invalid_disp = sgm.get_invalid_disparity();

	for (int frame_no = start_number;; frame_no++) {

		I1 = cv::imread(cv::format(image_format_L.c_str(), frame_no), cv::IMREAD_UNCHANGED);
		I2 = cv::imread(cv::format(image_format_R.c_str(), frame_no), cv::IMREAD_UNCHANGED);
		if (I1.empty() || I2.empty()) {
			frame_no = start_number - 1;
			continue;
		}

		d_I1.upload(I1.data);
		d_I2.upload(I2.data);

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		d_disparity.download(disparity.data);

		// draw results
		if (I1.type() != CV_8U)
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX, CV_8U);

		colorize_disparity(disparity, disparity_color, disp_size, disparity == invalid_disp);
		cv::putText(disparity_color, cv::format("sgm execution time: %4.1f[msec] %4.1f[FPS]",
			1e-3 * duration, fps), cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("left image", I1);
		cv::imshow("disparity", disparity_color);

		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}

	return 0;
}
