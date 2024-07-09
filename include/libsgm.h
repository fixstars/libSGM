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

#ifndef __LIBSGM_H__
#define __LIBSGM_H__

/**
* @mainpage stereo-sgm
* See sgm::StereoSGM
*/

/**
* @file libsgm.h
* stereo-sgm main header
*/

#include "libsgm_config.h"

#if defined(LIBSGM_SHARED)
#if defined(WIN32) || defined(_WIN32)
#if defined sgm_EXPORTS
#define LIBSGM_API __declspec(dllexport)
#else
#define LIBSGM_API __declspec(dllimport)
#endif
#else
#define LIBSGM_API __attribute__((visibility("default")))
#endif
#else
#define LIBSGM_API
#endif

namespace sgm
{

/**
* @brief Indicates input/output pointer type.
*/
enum ExecuteInOut
{
	EXECUTE_INOUT_HOST2HOST = (0 << 1) | 0,
	EXECUTE_INOUT_HOST2CUDA = (1 << 1) | 0,
	EXECUTE_INOUT_CUDA2HOST = (0 << 1) | 1,
	EXECUTE_INOUT_CUDA2CUDA = (1 << 1) | 1,
};

/**
* @brief Indicates number of scanlines which will be used.
*/
enum class PathType
{
	SCAN_4PATH, //>! Horizontal and vertical paths.
	SCAN_8PATH,  //>! Horizontal, vertical and oblique paths.
	SCAN_16PATH
};

/**
* @brief Indicates census type which will be used.
*/
enum class CensusType
{
	CENSUS_9x7,
	SYMMETRIC_CENSUS_9x7
};

/**
* @brief StereoSGM class
*/
class StereoSGM
{
public:

	static const int SUBPIXEL_SHIFT = 4;
	static const int SUBPIXEL_SCALE = (1 << SUBPIXEL_SHIFT);

	/**
	* @brief Available options for StereoSGM
	*/
	struct Parameters
	{
		int P1;
		int P2;
		float uniqueness;
		bool subpixel;
		PathType path_type;
		int min_disp;
		int LR_max_diff;
		CensusType census_type;

		/**
		* @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
		* @param P2 Penalty on the disparity change by more than 1 between neighbor pixels.
		* @param uniqueness Margin in ratio by which the best cost function value should be at least second one.
		* @param subpixel Disparity value has 4 fractional bits if subpixel option is enabled.
		* @param path_type Number of scanlines used in cost aggregation.
		* @param min_disp Minimum possible disparity value.
		* @param LR_max_diff Acceptable difference pixels which is used in LR check consistency. LR check consistency will be disabled if this value is set to negative.
		* @param census_type Type of census transform.
		*/
		LIBSGM_API Parameters(int P1 = 10, int P2 = 120, float uniqueness = 0.95f, bool subpixel = false, PathType path_type = PathType::SCAN_8PATH,
			int min_disp = 0, int LR_max_diff = 1, CensusType census_type = CensusType::SYMMETRIC_CENSUS_9x7);
	};

	/**
	* @param width Processed image's width.
	* @param height Processed image's height.
	* @param disparity_size It must be 64, 128 or 256.
	* @param input_depth_bits Processed image's bits per pixel. It must be 8, 16 or 32.
	* @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
	* @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
	* @attention
	* output_depth_bits must be set to 16 when subpixel is enabled.
	*/
	LIBSGM_API StereoSGM(int width, int height, int disparity_size, int input_depth_bits, int output_depth_bits,
		ExecuteInOut inout_type, const Parameters& param = Parameters());

	/**
	* @param width Processed image's width.
	* @param height Processed image's height.
	* @param disparity_size It must be 64, 128 or 256.
	* @param input_depth_bits Processed image's bits per pixel. It must be 8, 16 or 32.
	* @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
	* @param src_pitch Source image's pitch (pixels).
	* @param dst_pitch Destination image's pitch (pixels).
	* @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
	* @attention
	* output_depth_bits must be set to 16 when subpixel is enabled.
	*/
	LIBSGM_API StereoSGM(int width, int height, int disparity_size, int input_depth_bits, int output_depth_bits, int src_pitch, int dst_pitch,
		ExecuteInOut inout_type, const Parameters& param = Parameters());

	LIBSGM_API virtual ~StereoSGM();

	/**
	* Execute stereo semi global matching.
	* @param left_pixels  A pointer stored input left image.
	* @param right_pixels A pointer stored input right image.
	* @param dst          Output pointer. User must allocate enough memory.
	* @attention
	* You need to allocate dst memory at least width x height x sizeof(element_type) bytes.
	* The element_type is uint8_t for output_depth_bits == 8 and uint16_t for output_depth_bits == 16.
	* Note that dst element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
	* Value of Invalid disparity is equal to return value of `get_invalid_disparity` member function.
	*/
	LIBSGM_API void execute(const void* left_pixels, const void* right_pixels, void* dst);

	/**
	* Generate invalid disparity value from Parameter::min_disp and Parameter::subpixel
	* @attention
	* Cast properly if you receive disparity value as `unsigned` type.
	* See sample/movie for an example of this.
	*/
	LIBSGM_API int get_invalid_disparity() const;

private:

	StereoSGM(const StereoSGM&);
	StereoSGM& operator=(const StereoSGM&);

	class Impl;
	Impl* impl_;
};

} // namespace sgm

#endif // !__LIBSGM_H__

#include "libsgm_wrapper.h"
