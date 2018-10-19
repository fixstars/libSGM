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
#include <iomanip>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sl/Camera.hpp>

using namespace sl;

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

int main(int argc, char* argv[]) {	
	if (argc < 3) {
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int disp_size = 64;
	
	Camera zed;
	InitParameters initParameters;
	initParameters.camera_resolution = RESOLUTION_VGA;
	ERROR_CODE err = zed.open(initParameters);
	if (err != SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}
	const int width = zed.getResolution().width;
	const int height = zed.getResolution().height;

	// sl::Mat and cv::Mat share data over memory
	Mat zed_image_r(zed.getResolution(), MAT_TYPE_8U_C1);
	Mat zed_image_l(zed.getResolution(), MAT_TYPE_8U_C1);
	cv::Mat ocv_image_r(height, width, CV_8UC1, zed_image_r.getPtr<uchar>(MEM_CPU));
	cv::Mat ocv_image_l(height, width, CV_8UC1, zed_image_l.getPtr<uchar>(MEM_CPU));

	for (int frame_no = 1;; frame_no++) {
		if (zed.grab() == SUCCESS) {
			zed.retrieveImage(zed_image_l, VIEW_LEFT_UNRECTIFIED_GRAY, MEM_CPU);
			zed.retrieveImage(zed_image_r, VIEW_RIGHT_UNRECTIFIED_GRAY, MEM_CPU);
		} else continue;
		if (ocv_image_r.empty() || ocv_image_l.empty()) continue;
		imwrite(format_string(argv[1], frame_no), ocv_image_l);
		imwrite(format_string(argv[2], frame_no), ocv_image_r);
		
		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}
	zed.close();
	return 0;
}
