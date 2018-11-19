#ifndef __LIBSGM_WRAPPER_H__
#define __LIBSGM_WRAPPER_H__

#include "libsgm.h"
#include <memory>
#ifdef WITH_OPENCV
#include <opencv2/core/cuda.hpp>
#endif

namespace sgm {
	class LibSGMWrapper {
	public:
		static constexpr size_t DISPARITY_SIZE = 128;
		static constexpr int OUTPUT_DEPTH_BITS = 16;
		LibSGMWrapper(const sgm::StereoSGM::Parameters& param = {});
		~LibSGMWrapper();

#ifdef WITH_OPENCV
		void execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity);
		void execute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& disparity);
#endif // WITH_OPENCV

	private:
		struct Info;
		std::unique_ptr<sgm::StereoSGM> sgm_;
		sgm::StereoSGM::Parameters param_;
		std::unique_ptr<Info> prev_;
	};
}

#endif // __LIBSGM_WRAPPER_H__
