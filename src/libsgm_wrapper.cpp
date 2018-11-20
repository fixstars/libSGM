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
		sgm::EXECUTE_INOUT inout_type;

		bool operator==(const sgm::LibSGMWrapper::Info& rhs) const {
			return
				width == rhs.width
				&& height == rhs.height
				&& src_pitch == rhs.src_pitch
				&& dst_pitch == rhs.dst_pitch
				&& input_depth_bits == rhs.input_depth_bits
				&& inout_type == rhs.inout_type;
		}
		bool operator!=(const sgm::LibSGMWrapper::Info& rhs) const {
			return !(*this == rhs);
		}
	};

#ifdef WITH_OPENCV
	void LibSGMWrapper::execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity) {
		const cv::Size size = I1.size();
		CV_Assert(size == I2.size());
		CV_Assert(I1.type() == I2.type());
		const int depth = I1.depth();
		CV_Assert(depth == CV_8U || depth == CV_16U);
		if (disparity.size() != size || disparity.depth() != CV_16U) {
			disparity.create(size, CV_16U);
		}
		std::unique_ptr<Info> info(new Info());
		info->width = size.width;
		info->height = size.height;
		info->src_pitch = static_cast<int>(I1.step1());
		info->dst_pitch = static_cast<int>(disparity.step1());
		info->input_depth_bits = static_cast<int>(I1.elemSize1()) * 8;
		info->inout_type = sgm::EXECUTE_INOUT_CUDA2CUDA;
		if (!sgm_ || !prev_ || *info != *prev_) {
			sgm_.reset(new StereoSGM(info->width, info->height, DISPARITY_SIZE, info->input_depth_bits, OUTPUT_DEPTH_BITS, info->src_pitch, info->dst_pitch, info->inout_type, param_));
		}
		prev_ = std::move(info);

		sgm_->execute(I1.data, I2.data, disparity.data);
	}
	void LibSGMWrapper::execute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& disparity) {
		const cv::Size size = I1.size();
		CV_Assert(size == I2.size());
		CV_Assert(I1.type() == I2.type());
		const int depth = I1.depth();
		CV_Assert(depth == CV_8U || depth == CV_16U);
		if (disparity.size() != size || disparity.depth() != CV_16U) {
			disparity.create(size, CV_16U);
		}
		std::unique_ptr<Info> info(new Info());
		info->width = size.width;
		info->height = size.height;
		info->src_pitch = static_cast<int>(I1.step1());
		info->dst_pitch = static_cast<int>(disparity.step1());
		info->input_depth_bits = static_cast<int>(I1.elemSize1()) * 8;
		info->inout_type = sgm::EXECUTE_INOUT_HOST2HOST;
		if (!sgm_ || !prev_ || *info != *prev_) {
			sgm_.reset(new StereoSGM(info->width, info->height, DISPARITY_SIZE, info->input_depth_bits, OUTPUT_DEPTH_BITS, info->src_pitch, info->dst_pitch, info->inout_type, param_));
		}
		prev_ = std::move(info);

		sgm_->execute(I1.data, I2.data, disparity.data);
	}
#endif // WITH_OPENCV
}
