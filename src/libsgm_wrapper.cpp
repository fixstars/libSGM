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
	LibSGMWrapper::LibSGMWrapper(int numDisparity, int P1, int P2, float uniquenessRatio, bool subpixel)
		: sgm_(nullptr), numDisparity_(numDisparity), param_(P1, P2, uniquenessRatio, subpixel), prev_(nullptr) {}
	LibSGMWrapper::~LibSGMWrapper() = default;

	int LibSGMWrapper::getNumDisparities() const { return numDisparity_; }
	float LibSGMWrapper::getUniquenessRatio() const { return param_.uniqueness; }
	int LibSGMWrapper::getP1() const { return param_.P1; }
	int LibSGMWrapper::getP2() const { return param_.P2; }
	bool LibSGMWrapper::hasSubpixel() const { return param_.subpixel; }

	struct LibSGMWrapper::Creator {
		int width;
		int height;
		int src_pitch;
		int dst_pitch;
		int input_depth_bits;
		int output_depth_bits;
		sgm::EXECUTE_INOUT inout_type;

		bool operator==(const Creator& rhs) const {
			return
				width == rhs.width
				&& height == rhs.height
				&& src_pitch == rhs.src_pitch
				&& dst_pitch == rhs.dst_pitch
				&& input_depth_bits == rhs.input_depth_bits
				&& output_depth_bits == rhs.output_depth_bits
				&& inout_type == rhs.inout_type;
		}
		bool operator!=(const Creator& rhs) const {
			return !(*this == rhs);
		}

		StereoSGM* createStereoSGM(int disparity_size, const StereoSGM::Parameters& param) {
			return new StereoSGM(width, height, disparity_size, input_depth_bits, output_depth_bits, src_pitch, dst_pitch, inout_type, param);
		}

#ifdef BUILD_OPENCV_WRAPPER
		Creator(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& dst) {
			const int depth = src.depth();
			CV_Assert(depth == CV_8U || depth == CV_16U);
			width = src.cols;
			height = src.rows;
			src_pitch = static_cast<int>(src.step1());
			dst_pitch = static_cast<int>(dst.step1());
			input_depth_bits = static_cast<int>(src.elemSize1()) * 8;
			output_depth_bits = static_cast<int>(dst.elemSize1()) * 8;
			inout_type = sgm::EXECUTE_INOUT_CUDA2CUDA;
		}
		Creator(const cv::Mat& src, const cv::Mat& dst) {
			const int depth = src.depth();
			CV_Assert(depth == CV_8U || depth == CV_16U);
			width = src.cols;
			height = src.rows;
			src_pitch = static_cast<int>(src.step1());
			dst_pitch = static_cast<int>(dst.step1());
			input_depth_bits = static_cast<int>(src.elemSize1()) * 8;
			output_depth_bits = static_cast<int>(dst.elemSize1()) * 8;
			inout_type = sgm::EXECUTE_INOUT_HOST2HOST;
		}
#endif // BUILD_OPRENCV_WRAPPER
	};

#ifdef BUILD_OPENCV_WRAPPER
	void LibSGMWrapper::execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity) {
		const cv::Size size = I1.size();
		CV_Assert(size == I2.size());
		CV_Assert(I1.type() == I2.type());
		const int depth = I1.depth();
		CV_Assert(depth == CV_8U || depth == CV_16U);
		if (disparity.size() != size || disparity.depth() != CV_16U) {
			disparity.create(size, CV_16U);
		}
		std::unique_ptr<Creator> creator(new Creator(I1, disparity));
		if (!sgm_ || !prev_ || *creator != *prev_) {
			sgm_.reset(creator->createStereoSGM(numDisparity_, param_));
		}
		prev_ = std::move(creator);

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
		std::unique_ptr<Creator> creator(new Creator(I1, disparity));
		if (!sgm_ || !prev_ || *creator != *prev_) {
			sgm_.reset(creator->createStereoSGM(numDisparity_, param_));
		}
		prev_ = std::move(creator);

		sgm_->execute(I1.data, I2.data, disparity.data);
	}
#endif // BUILD_OPENCV_WRAPPER
}
