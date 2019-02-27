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
#include <opencv2/core/version.hpp>

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

static void execute(sgm::LibSGMWrapper& sgmw, const cv::Mat& h_left, const cv::Mat& h_right, cv::Mat& h_disparity) noexcept(false)
{
	cv::cuda::GpuMat d_left, d_right;
	d_left.upload(h_left);
	d_right.upload(h_right);

	cv::cuda::GpuMat d_disparity;

	sgmw.execute(d_left, d_right, d_disparity);

	// normalize result
	cv::cuda::GpuMat d_normalized_disparity;
	d_disparity.convertTo(d_normalized_disparity, CV_8UC1, 256. / sgmw.getNumDisparities());
	d_normalized_disparity.download(h_disparity);
}

int main(int argc, char* argv[]) {
	ASSERT_MSG(argc >= 3, "usage: stereosgm left_img right_img [disp_size]");

	const cv::Mat left = cv::imread(argv[1], -1);
	const cv::Mat right = cv::imread(argv[2], -1);
	const int disp_size = argc > 3 ? std::atoi(argv[3]) : 128;

	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

	sgm::LibSGMWrapper sgmw{disp_size};
	cv::Mat disparity;
	try {
		execute(sgmw, left, right, disparity);
	} catch (const cv::Exception& e) {
		std::cerr << e.what() << std::endl;
		if (e.code == cv::Error::GpuNotSupported) {
			return 1;
		} else {
			return -1;
		}
	}

	// post-process for showing image
	cv::Mat colored;
	cv::applyColorMap(disparity, colored, cv::COLORMAP_JET);
	cv::imshow("image", disparity);

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
					cv::imshow("image", disparity);
					break;
				}
			case 1:
				{
					cv::setWindowTitle("image", "disparity color");
					cv::imshow("image", colored);
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
