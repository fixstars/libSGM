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

#include <libsgm_wrapper.h>

namespace sgm {
	LibSGMWrapper::LibSGMWrapper(const sgm::StereoSGM::Parameters& param) : sgm_(nullptr), param_(param), prev_(nullptr) {}
	LibSGMWrapper::~LibSGMWrapper() = default;

	struct LibSGMWrapper::Info {
		int width;
		int height;
		int src_pitch;
		int dst_pitch;
		int input_depth_bits;
	};

#ifdef WITH_OPENCV
	void LibSGMWrapper::execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity) {
		// TODO: implement
	}

#endif // WITH_OPENCV
}
