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
		/**
		 * @param numDisparity Maximum disparity minus minimum disparity
		 * @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels
		 * @param P2 Penalty on the disparity change by more than 1 between neighbor pixels
		 * @param uniquenessRatio Margin in ratio by which the best cost function value should be at least second one
		 * @param subpixel Disparity value has 4 fractional bits if subpixel option is enabled
		 * @param pathType Number of scanlines used in cost aggregation
		 * @param minDisparity Minimum possible disparity value
		 */
		LIBSGM_API LibSGMWrapper(int numDisparity = 128, int P1 = 10, int P2 = 120, float uniquenessRatio = 0.95f,
				bool subpixel = false, PathType pathType = PathType::SCAN_8PATH, int minDisparity = 0);
		LIBSGM_API ~LibSGMWrapper();

		LIBSGM_API int getNumDisparities() const;
		LIBSGM_API int getP1() const;
		LIBSGM_API int getP2() const;
		LIBSGM_API float getUniquenessRatio() const;
		LIBSGM_API bool hasSubpixel() const;
		LIBSGM_API PathType getPathType() const;
		LIBSGM_API int getMinDisparity() const;
		LIBSGM_API int getInvalidDisparity() const;

#ifdef BUILD_OPENCV_WRAPPER
		/**
		 * Execute stereo semi global matching via wrapper class
		 * @param I1        Input left image.  Image's type is must be CV_8U or CV_16U
		 * @param I2        Input right image.  Image's size and type must be same with I1.
		 * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
		 * @attention
		 * type of output image `disparity` is CV_16S.
		 * Note that disparity element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
		 */
		LIBSGM_API void execute(const cv::cuda::GpuMat& I1, const cv::cuda::GpuMat& I2, cv::cuda::GpuMat& disparity);
		/**
		 * Execute stereo semi global matching via wrapper class
		 * @param I1        Input left image.  Image's type is must be CV_8U or CV_16U
		 * @param I2        Input right image.  Image's size and type must be same with I1.
		 * @param disparity Output image.  Its memory will be allocated automatically dependent on input image size.
		 * @attention
		 * type of output image `disparity` is CV_16S.
		 * Note that disparity element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
		 */
		LIBSGM_API void execute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& disparity);
#endif // BUILD_OPRENCV_WRAPPER

	private:
		struct Creator;
		std::unique_ptr<sgm::StereoSGM> sgm_;
		int numDisparity_;
		sgm::StereoSGM::Parameters param_;
		std::unique_ptr<Creator> prev_;
	};
}

#endif // __LIBSGM_WRAPPER_H__
