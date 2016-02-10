#pragma once

/**
* @mainpage  stereo-sgm
*
* See sgm::StereoSGM
*/

/**
* @file libsgm.h
* @brief stereo-sgm main header
*/

namespace sgm {
	struct CudaStereoSGMResources;

	/**
	* @enum DST_TYPE
	* @brief Indicates destination type.
	*/
	enum DST_TYPE {
		/// Return host pointer. It must not free or delete.
		DST_TYPE_HOST_PTR, 
		/// Return cuda pointer. It must not free.
		DST_TYPE_CUDA_PTR, 
	};

	/**
	* @brief StereoSGM class
	*/
	class StereoSGM {
	public:
		/**
		* @param width Processed image's width. It must be even.
		* @param height Processed image's height. It must be even.
		* @param depth_bits Processed image's bits per pixel. It must be 16 now.
		* @param disparity_size It must be 64 or 128.
		*/
		StereoSGM(int width, int height, int depth_bits, int disparity_size);

		virtual ~StereoSGM();

		/**
		* @brief Execute stereo semi global matching.
		* @param left_pixels A pointer stored input left image.
		* @param right_pixels A pointer stored input rigth image.
		* @param dst Output pointer. User must allocate enoght memory.
		* @param dst_type Specify output pointer type. See #sgm::DST_TYPE.
		* @param depth_bits bits per pixel. It must be 8 or 16.
		* 
		* @note
		* For example, when dst_type == DST_TYPE_HOST_PTR, depth_bits == 8, allocate memory with 
		* @code 
		* uint8_t* dst = new uint8_t[image_width * image_height];
		* @endcode
		* @attention 
		* For performance reason, when dst_type == DST_TYPE_CUDA_PTR, depth_bits == 16, you don't have to allocate dst memory yourself. It returns internal cuda pointer. You must not free the pointer.
		*/
		void execute(const void* left_pixels, const void* right_pixels, void** dst, DST_TYPE dst_type, int depth_bits = 16);
		
	private:
		StereoSGM(const StereoSGM&);
		StereoSGM& operator=(const StereoSGM&);

		void cuda_resource_allocate();

		CudaStereoSGMResources* cu_res_;

		int width_;
		int height_;
		int depth_bits_;
		int disparity_size_;
	};
}
