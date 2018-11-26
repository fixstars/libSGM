#ifndef __LIBSGM_WRAPPER_H__
#define __LIBSGM_WRAPPER_H__

#include "libsgm.h"
#include <memory>
#ifdef BUILD_OPENCV_WRAPPER
#include <opencv2/core/cuda.hpp>
#endif

namespace sgm {
	/**
	 * @brief LibSGMWrapper class which is wrapper for sgm::StereoSGM.
	 */
	class LibSGMWrapper {
	public:
		static constexpr size_t DISPARITY_SIZE = 128;
		static constexpr int OUTPUT_DEPTH_BITS = 16;
		/**
		 * @param param You can specify parameters.  See libsgm.h for more information.
		 */
		LIBSGM_API LibSGMWrapper(const sgm::StereoSGM::Parameters& param = {});
		LIBSGM_API ~LibSGMWrapper();

#ifdef BUILD_OPENCV_WRAPPER
		/**
		 * Execute stereo semi global matching via wrapper class
		 * @param I1        Input left image.  Image's type is must be CV_8U or CV_16U
		 * @param I2        Input right image.  Image's size and type must be same with I1.
		 * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
		 * @attention
		 * type of output image `disparity` is CV_16U.
		 * Note that dst element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
		 */
		LIBSGM_API void execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity);
		/**
		 * Execute stereo semi global matching via wrapper class
		 * @param I1        Input left image.  Image's type is must be CV_8U or CV_16U
		 * @param I2        Input right image.  Image's size and type must be same with I1.
		 * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
		 * @attention
		 * type of output image `disparity` is CV_16U.
		 * Note that dst element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
		 */
		LIBSGM_API void execute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& disparity);
#endif // BUILD_OPRENCV_WRAPPER

	private:
		struct Info;
		std::unique_ptr<sgm::StereoSGM> sgm_;
		sgm::StereoSGM::Parameters param_;
		std::unique_ptr<Info> prev_;
	};
}

#endif // __LIBSGM_WRAPPER_H__
