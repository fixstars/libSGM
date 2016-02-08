#pragma once

namespace sgm {
	struct CudaStereoSGMResources;

	enum DST_TYPE {
		DST_TYPE_HOST_PTR,
		DST_TYPE_CUDA_PTR,
	};

	class StereoSGM {
	public:
		StereoSGM(int width, int height, int depth_bits, int disparity_size);
		virtual ~StereoSGM();

		void execute(const void* left_pixels, const void* right_pixels, void** dst, DST_TYPE dst_type, int depth_bits = 16);

		//void get_image(void** dst, DST_TYPE dst_type, int depth_bits = 16, DST_OUTPUT_SIDE dst_side = DST_OUTPUT_SIDE_LEFT);

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
