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
#include <opencv2/core/cuda.hpp>
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
	ASSERT_MSG(argc >= 3, "usage: stereosgm left_img right_img [disp_size]");

	cv::Mat  left = cv::imread(argv[1], -1);
	cv::Mat right = cv::imread(argv[2], -1);
	const int disp_size = argc > 3 ? std::atoi(argv[3]) : 128;

	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

	sgm::LibSGMWrapper sgm(disp_size);
	cv::Mat disparity;

	try {
		cv::cuda::GpuMat d_left, d_right, d_disparity;
		d_left.upload(left);
		d_right.upload(right);
		sgm.execute(d_left, d_right, d_disparity);
		d_disparity.download(disparity);
	}
	catch (const cv::Exception& e) {
		std::cerr << e.what() << std::endl;
		return e.code == cv::Error::GpuNotSupported ? 1 : -1;
	}

	// show image
	cv::Mat mask = disparity == sgm.getInvalidDisparity();
	cv::Mat disparity_color;
	disparity.convertTo(disparity, CV_8U, 255. / disp_size);
	cv::applyColorMap(disparity, disparity_color, cv::COLORMAP_JET);
	disparity.setTo(0, mask);
	disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
	if (left.type() != CV_8U)
		cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8U);

	std::vector<cv::Mat> images = { disparity, disparity_color, left };
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
