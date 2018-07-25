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
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libsgm.h>

#include "demo.h"
#include "renderer.h"

int main(int argc, char* argv[]) {

	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img_fmt right_img_fmt [disp_size] [max_frame_num]" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::string left_filename_fmt, right_filename_fmt;
	left_filename_fmt = argv[1];
	right_filename_fmt = argv[2];

	// dangerous
	char buf[1024];
	sprintf(buf, left_filename_fmt.c_str(), 0);
	cv::Mat left = cv::imread(buf, -1);
	sprintf(buf, right_filename_fmt.c_str(), 0);
	cv::Mat right = cv::imread(buf, -1);


	int disp_size = 64;
	if (argc >= 4) {
		disp_size = atoi(argv[3]);
	}

	int max_frame = 100;
	if(argc >= 5) {
		max_frame = atoi(argv[4]);
	}


	if (left.size() != right.size() || left.type() != right.type()) {
		std::cerr << "mismatch input image size" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int bits = 0;

	switch (left.type()) {
	case CV_8UC1: bits = 8; break;
	case CV_16UC1: bits = 16; break;
	default:
		std::cerr << "invalid input image color format" << left.type() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int width = left.cols;
	int height = left.rows;

	cudaGLSetGLDevice(0);

	SGMDemo demo(width, height);

	if (demo.init()) {
		printf("fail to init SGM Demo\n");
		std::exit(EXIT_FAILURE);
	}

	sgm::StereoSGM ssgm(width, height, disp_size, bits, 16, sgm::EXECUTE_INOUT_HOST2CUDA);

	Renderer renderer(width, height);
	
	uint16_t* d_output_buffer = NULL;
	cudaMalloc((void**)&d_output_buffer, sizeof(uint16_t) * width * height);

	int frame_no = 0;
	while (!demo.should_close()) {

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		
		if (frame_no == max_frame) { frame_no = 0; }

		sprintf(buf, left_filename_fmt.c_str(), frame_no);
		cv::Mat left = cv::imread(buf, -1);
		sprintf(buf, right_filename_fmt.c_str(), frame_no);
		cv::Mat right = cv::imread(buf, -1);

		if (left.size() == cv::Size(0, 0) || right.size() == cv::Size(0, 0)) {
			max_frame = frame_no;
			frame_no = 0;
			continue;
		}

		ssgm.execute(left.data, right.data, d_output_buffer);

		switch (demo.get_flag()) {
		case 0:
			{
				renderer.render_input((uint16_t*)left.data);
			}
			break;
		case 1:
			renderer.render_disparity(d_output_buffer, disp_size);
			break;
		case 2:
			renderer.render_disparity_color(d_output_buffer, disp_size);
			break;
		}
		
		demo.swap_buffer();
		frame_no++;
	}

	cudaFree(d_output_buffer);
}
