/*
Copyright 2016 fixstars

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
