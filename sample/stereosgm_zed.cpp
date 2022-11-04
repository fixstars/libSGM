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

#include <sl/Camera.hpp>

#include <libsgm.h>

#include "sample_common.h"

static const std::string keys =
"{disp_size           |    128 | maximum possible disparity value                  }"
"{camera_resolution   |      3 | camera resolution (0:HD2K 1:HD1080 2:HD720 3:VGA) }"
"{help h              |        | display this help and exit                        }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	const int disp_size = parser.get<int>("disp_size");
	const sl::RESOLUTION camera_resolution = parser.get<sl::RESOLUTION>("camera_resolution");

	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = camera_resolution;
	const sl::ERROR_CODE err = zed.open(initParameters);
	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cerr << sl::toString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const auto& resolution = zed.getCameraInformation().camera_configuration.resolution;
	sl::Mat d_zed_image_L(resolution, sl::MAT_TYPE::U8_C1, sl::MEM::GPU);
	sl::Mat d_zed_image_R(resolution, sl::MAT_TYPE::U8_C1, sl::MEM::GPU);
	CV_Assert(d_zed_image_L.getStep(sl::MEM::GPU) == d_zed_image_R.getStep(sl::MEM::GPU));

	const int width = resolution.width;
	const int height = resolution.height;
	const int src_pitch = static_cast<int>(d_zed_image_L.getStep(sl::MEM::GPU));
	const int dst_pitch = width;

	const int src_depth = 8;
	const int dst_depth = disp_size < 256 ? 8 : 16;
	const int src_bytes = src_depth * width * height / 8;
	const int dst_bytes = dst_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, src_pitch, dst_pitch, sgm::EXECUTE_INOUT_CUDA2CUDA);

	device_buffer d_disparity(dst_bytes);
	cv::Mat disparity(height, width, dst_depth == 8 ? CV_8S : CV_16S), disparity_color;

	const int invalid_disp = sgm.get_invalid_disparity();

	std::cout << "max disparity    : " << disp_size << std::endl;
	std::cout << "camera resolution: " << sl::toString(initParameters.camera_resolution) << " " << cv::Size(width, height) << std::endl;

	while (1) {

		if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
			zed.retrieveImage(d_zed_image_L, sl::VIEW::LEFT_GRAY, sl::MEM::GPU);
			zed.retrieveImage(d_zed_image_R, sl::VIEW::RIGHT_GRAY, sl::MEM::GPU);
		}
		else {
			continue;
		}

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_zed_image_L.getPtr<uchar>(sl::MEM::GPU), d_zed_image_R.getPtr<uchar>(sl::MEM::GPU), d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		d_disparity.download(disparity.data);

		// draw results
		colorize_disparity(disparity, disparity_color, disp_size, disparity == invalid_disp);
		cv::putText(disparity_color, cv::format("sgm execution time: %4.1f[msec] %4.1f[FPS]",
			1e-3 * duration, fps), cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("disparity", disparity_color);

		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}

	return 0;
}
