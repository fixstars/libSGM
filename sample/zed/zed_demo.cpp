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

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sl/Camera.hpp>

#include <libsgm.h>

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

struct device_buffer {
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

int main(int argc, char* argv[]) {	
	
	const int disp_size = 128;
	
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION_VGA;
	sl::ERROR_CODE err = zed.open(initParameters);
	if (err != sl::SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}
	const int width = static_cast<int>(zed.getResolution().width);
	const int height = static_cast<int>(zed.getResolution().height);

	// sl::Mat and cv::Mat share data over memory
	sl::Mat zed_image_r(zed.getResolution(), sl::MAT_TYPE_8U_C1);
	sl::Mat zed_image_l(zed.getResolution(), sl::MAT_TYPE_8U_C1);
	cv::Mat ocv_image_r(height, width, CV_8UC1, zed_image_r.getPtr<uchar>(sl::MEM_CPU));
	cv::Mat ocv_image_l(height, width, CV_8UC1, zed_image_l.getPtr<uchar>(sl::MEM_CPU));

	const int input_depth = 8;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = 8;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	cv::Mat disparity(height, width, CV_8U);
	cv::Mat disparity_8u, disparity_color;

	device_buffer d_image_r(input_bytes), d_image_l(input_bytes), d_disparity(output_bytes);

	while (1) {
		if (zed.grab() == sl::SUCCESS) {
			zed.retrieveImage(zed_image_l, sl::VIEW_LEFT_GRAY, sl::MEM_CPU);
			zed.retrieveImage(zed_image_r, sl::VIEW_RIGHT_GRAY, sl::MEM_CPU);
		} else continue;
		if (ocv_image_r.empty() || ocv_image_l.empty()) continue;

		cudaMemcpy(d_image_r.data, ocv_image_r.data, input_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_image_l.data, ocv_image_l.data, input_bytes, cudaMemcpyHostToDevice);
		
		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_image_l.data, d_image_r.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("input left", ocv_image_l);
		cv::imshow("disparity", disparity_color);
		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}
	zed.close();
	return 0;
}
