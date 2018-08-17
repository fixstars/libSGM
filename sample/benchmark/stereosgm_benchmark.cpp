/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unlestd::cout required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exprestd::cout or implied.
See the License for the specific language governing permistd::coutions and
limitations under the License.
*/

#include <iostream>
#include <iomanip>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

int main(int argc, char* argv[])
{
	if (argc < 3) {
		std::cout << "usage: " << argv[0] << " left_img right_img [disp_size] [iterations]" << std::endl;
		return 0;
	}

	cv::Mat I1 = cv::imread(argv[1], -1);
	cv::Mat I2 = cv::imread(argv[2], -1);

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");

	const int disp_size = argc > 3 ? std::stoi(argv[3]) : 128;
	const int iterations = argc > 4 ? std::stoi(argv[4]) : 100;

	const int width = I1.cols;
	const int height = I1.rows;

	const int input_depth = I1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int out_depth = 8;
	const int output_bytes = out_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, out_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
	cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

	cudaDeviceProp prop;
	int version;
	cudaGetDeviceProperties(&prop, 0);
	cudaRuntimeGetVersion(&version);

	// show settings
	std::cout << "# Settings" << std::endl;
	std::cout << "device name         : " << prop.name << std::endl;
	std::cout << "CUDA runtime version: " << version << std::endl;
	std::cout << "image size          : " << I1.size() << std::endl;
	std::cout << "disparity size      : " << disp_size << std::endl;
	std::cout << "sgm path            : " << "8 path" << std::endl;
	std::cout << "iterations          : " << iterations << std::endl;
	std::cout << std::endl;

	// run benchmark
	std::cout << "Running benchmark..." << std::endl;
	uint64_t sum = 0;
	for (int i = 0; i <= iterations; i++) {
		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		if (i > 0)
			sum += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	}
	std::cout << "Done." << std::endl << std::endl;

	// show results
	const double time_millisec = 1e-3 * sum / iterations;
	const double fps = 1e3 / time_millisec;
	std::cout << "# Results" << std::endl;
	std::cout.setf(std::ios::fixed);
	std::cout << std::setprecision(1) << "Processing Time[Milliseconds]: " << time_millisec << std::endl;
	std::cout << std::setprecision(1) << "FPS                          : " << fps << std::endl;
	std::cout << std::endl;

	// save disparity image
	cv::Mat disparity(height, width, CV_8U);
	cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
	disparity *= 255. / disp_size;
	cv::imwrite("disparity.png", disparity);

	return 0;
}