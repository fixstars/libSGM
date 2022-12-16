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

#include "sample_common.h"

#include <opencv2/imgproc.hpp>

void colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask)
{
	cv::Mat tmp;
	src.convertTo(tmp, CV_8U, 255. / disp_size);
	cv::applyColorMap(tmp, dst, cv::COLORMAP_TURBO);

	if (!mask.empty())
		dst.setTo(0, mask);
}
