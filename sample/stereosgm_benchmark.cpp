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
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libsgm.h>

#include "sample_common.h"

static const std::string keys =
"{ @left_img   | <none> | path to input left image                                       }"
"{ @right_img  | <none> | path to input right image                                      }"
"{ disp_size   |    128 | maximum possible disparity value                               }"
"{ out_depth   |      8 | disparity image's bits per pixel                               }"
"{ subpixel    |        | enable subpixel estimation                                     }"
"{ num_paths   |      8 | number of scanlines used in cost aggregation                   }"
"{ census_type |      1 | type of census transform (0:CENSUS_9x7 1:SYMMETRIC_CENSUS_9x7) }"
"{ iterations  |    100 | number of iterations for measuring performance                 }"
"{ help h      |        | display this help and exit                                     }";

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
	const int dst_depth = parser.get<int>("out_depth");
	const bool subpixel = parser.has("subpixel");
	const int num_paths = parser.get<int>("num_paths");
	const auto census_type = static_cast<sgm::CensusType>(parser.get<int>("census_type"));
	const int iterations = parser.get<int>("iterations");

	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8 || num_paths == 16, "number of scanlines must be 4, 8 or 16.");
	ASSERT_MSG(census_type == sgm::CensusType::CENSUS_9x7 || census_type == sgm::CensusType::SYMMETRIC_CENSUS_9x7, "census type must be 0 or 1.");
	ASSERT_MSG(dst_depth == 8 || dst_depth == 16, "output depth bits must be 8 or 16");
	if (subpixel)
		ASSERT_MSG(dst_depth == 16, "output depth bits must be 16 if subpixel option is enabled.");

	const int width = I1.cols;
	const int height = I1.rows;

	const int src_depth = I1.type() == CV_8U ? 8 : 16;
	const int src_bytes = src_depth * width * height / 8;
	const int dst_bytes = dst_depth * width * height / 8;
	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : num_paths == 4 ? sgm::PathType::SCAN_4PATH  : sgm::PathType::SCAN_16PATH;

	const sgm::StereoSGM::Parameters param(10, 120, 0.95f, subpixel, path_type, 0, 1, census_type);
	sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, param);

	device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
	cv::Mat disparity(height, width, dst_depth == 8 ? CV_8S : CV_16S);

	d_I1.upload(I1.data);
	d_I2.upload(I2.data);

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
	std::cout << "output depth        : " << dst_depth << std::endl;
	std::cout << "subpixel option     : " << (subpixel ? "true" : "false") << std::endl;
	std::cout << "sgm path            : " << num_paths << " path" << std::endl;
	std::cout << "census type         : " << (census_type == sgm::CensusType::CENSUS_9x7 ? "CENSUS_9x7" : "SYMMETRIC_CENSUS_9x7") << std::endl;
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
	const int disp_scale = subpixel ? sgm::StereoSGM::SUBPIXEL_SCALE : 1;
	d_disparity.download(disparity.data);
	colorize_disparity(disparity, disparity, disp_scale * disp_size, disparity == sgm.get_invalid_disparity());
	cv::imwrite("disparity.png", disparity);

	return 0;
}
