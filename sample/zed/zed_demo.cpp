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

#include <zed/Camera.hpp>

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
	auto cap = new sl::zed::Camera(sl::zed::ZEDResolution_mode::VGA);
	sl::zed::ERRCODE err = cap->init(sl::zed::MODE::PERFORMANCE, 0, true);
	if (err != sl::zed::ERRCODE::SUCCESS) {
		std::cout << sl::zed::errcode2str(err) << std::endl;
		exit(EXIT_FAILURE);
	}

	int width = cap->getImageSize().width;
	int height = cap->getImageSize().height;

	sgm::StereoSGM ssgm(width, height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2CUDA);

	SGMDemo demo(width, height);
	if (demo.init()) {
		printf("fail to init SGM Demo\n");
		std::exit(EXIT_FAILURE);
	}

	Renderer renderer(width, height);

	uint16_t* d_output_buffer = NULL;

	while (!demo.should_close()) {
		cap->grab(sl::zed::SENSING_MODE::FULL, false, false);

		sl::zed::Mat left_zm = cap->retrieveImage(sl::zed::SIDE::LEFT);
		
		cv::Mat left = cv::Mat(left_zm.height, left_zm.width, CV_8UC4, left_zm.data); // sl::zed::Mat to cv::Mat
		cv::cvtColor(left, left, CV_RGB2GRAY);

		sl::zed::Mat right_zm = cap->retrieveImage(sl::zed::SIDE::RIGHT);
		cv::Mat right = cv::Mat(right_zm.height, right_zm.width, CV_8UC4, right_zm.data); // sl::zed::Mat to cv::Mat
		cv::cvtColor(right, right, CV_RGB2GRAY);

		ssgm.execute(left.data, right.data, (void**)&d_output_buffer);

		switch (demo.get_flag()) {
		case 0: renderer.render_input((uint8_t*)left.data); break;
		case 1: renderer.render_disparity(d_output_buffer, disp_size); break;
		case 2: renderer.render_disparity_color(d_output_buffer, disp_size); break;
		}

		demo.swap_buffer();
	}
	delete cap;
}
