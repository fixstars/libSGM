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
#if CV_MAJOR_VERSION == 2
#include <opencv2/contrib/contrib.hpp>
#endif

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

static void execute(sgm::LibSGMWrapper& sgmw, const cv::Mat& _left, const cv::Mat& _right, cv::Mat& dst) noexcept(false)
{
	cv::cuda::GpuMat left, right;
	left.upload(_left);
	right.upload(_right);

	cv::cuda::GpuMat output;

	sgmw.execute(left, right, output);

	// normalize result
	cv::cuda::GpuMat processed;
	output.convertTo(processed, CV_8UC1, 256. / sgm::LibSGMWrapper::DISPARITY_SIZE);
	processed.download(dst);
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img right_img" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const cv::Mat left = cv::imread(argv[1], -1);
	const cv::Mat right = cv::imread(argv[2], -1);

	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");

	sgm::LibSGMWrapper sgmw;
	cv::Mat processed;
	try {
		execute(sgmw, left, right, processed);
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
	cv::applyColorMap(processed, colored, cv::COLORMAP_JET);
	cv::imshow("image", processed);

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
					cv::imshow("image", processed);
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

	return 0;
}
