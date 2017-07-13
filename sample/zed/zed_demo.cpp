/*
Copyright 2016 fixstars

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
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <nppi.h>

#include <sl/Camera.hpp>

#include <libsgm.h>

#include "demo.h"
#include "renderer.h"

int main(int argc, char* argv[]) {	
	
	int disp_size = 64;
	const int bits = 8;

	if (argc >= 2) {
		disp_size = atoi(argv[1]);
	}
	
	// init zed cam
	sl::Camera cap;
	sl::InitParameters init_param;
	init_param.camera_resolution = sl::RESOLUTION::RESOLUTION_HD1080;
	init_param.camera_fps = 15;
	sl::ERROR_CODE err = cap.open(init_param);
	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cout << sl::errorCode2str(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int width = cap.getResolution().width;
	int height = cap.getResolution().height;

	sgm::StereoSGM ssgm(width, height, disp_size, 8, 16, sgm::EXECUTE_INOUT_CUDA2CUDA);

	SGMDemo demo(width, height);
	if (demo.init()) {
		printf("fail to init SGM Demo\n");
		std::exit(EXIT_FAILURE);
	}

	Renderer renderer(width, height);

	uint16_t* d_output_buffer = NULL;
	uint8_t* d_input_left = NULL;
	uint8_t* d_input_right = NULL;
	cudaMalloc((void**)&d_input_left, width * height);
	cudaMalloc((void**)&d_input_right, width * height);

	const NppiSize roi = { width, height };

	cv::Mat h_input_left(height, width, CV_8UC1);

	while (!demo.should_close()) {
		err = cap.grab({sl::SENSING_MODE::SENSING_MODE_STANDARD, false, false, false})
		if (err != sl::ERROR_CODE::SUCCESS) {
			std::cout << sl::errorCode2str(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		sl::Mat left_zm;
		err = cap.retrieveImage(left_zm, sl::VIEW::VIEW_LEFT, sl::MEM::MEM_GPU);
		if (err != sl::ERROR_CODE::SUCCESS) {
			std::cout << sl::errorCode2str(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		sl::Mat right_zm;
		err = cap.retrieveImage(right_zm, sl::VIEW::VIEW_RIGHT, sl::MEM::MEM_GPU);
		if (err != sl::ERROR_CODE::SUCCESS) {
			std::cout << sl::errorCode2str(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		nppiRGBToGray_8u_AC4C1R(left_zm.getPtr<uint8_t>(sl::MEM::MEM_GPU), width * 4, d_input_left, width, roi);
		nppiRGBToGray_8u_AC4C1R(right_zm.getPtr<uint8_t>(sl::MEM::MEM_GPU), width * 4, d_input_right, width, roi);

		ssgm.execute(d_input_left, d_input_right, (void**)&d_output_buffer);

		switch (demo.get_flag()) {
		case 0: 
			cudaMemcpy(h_input_left.data, d_input_left, width * height, cudaMemcpyDeviceToHost);
			renderer.render_input((uint8_t*)h_input_left.data); 
			break;
		case 1: 
			renderer.render_disparity(d_output_buffer, disp_size);
			break;
		case 2: 
			renderer.render_disparity_color(d_output_buffer, disp_size);
			break;
		}

		demo.swap_buffer();
	}

	cudaFree(d_input_left);
	cudaFree(d_input_right);
	cap.close();
}
