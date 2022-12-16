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

#ifndef __SAMPLE_COMMON_H__
#define __SAMPLE_COMMON_H__

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#define ASSERT_MSG(expr, msg) \
if (!(expr)) { \
	std::cerr << msg << std::endl; \
	std::exit(EXIT_FAILURE); \
} \

struct device_buffer
{
	device_buffer() : data(nullptr), size(0) {}
	device_buffer(size_t count) : device_buffer() { allocate(count); }
	~device_buffer() { cudaFree(data); }

	void allocate(size_t count) { cudaMalloc(&data, count); size = count; }
	void upload(const void* h_data) { cudaMemcpy(data, h_data, size, cudaMemcpyHostToDevice); }
	void download(void* h_data) { cudaMemcpy(h_data, data, size, cudaMemcpyDeviceToHost); }

	void* data;
	size_t size;
};

void colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask = cv::noArray());

#endif // !__SAMPLE_COMMON_H__
