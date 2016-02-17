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

#include "internal.h"

namespace {
	template<typename SRC_T>
	__global__ void check_consistency_kernel(uint16_t* d_leftDisp, const uint16_t* d_rightDisp, const SRC_T* d_left, int width, int height)  {

		const int j = blockIdx.x * blockDim.x + threadIdx.x;
		const int i = blockIdx.y * blockDim.y + threadIdx.y;

		// left-right consistency check, only on leftDisp, but could be done for rightDisp too

		SRC_T mask = d_left[i * width + j];
		int d = d_leftDisp[i * width + j];
		int k = j - d;
		if (mask == 0 || d <= 0 || (k >= 0 && k < width && abs(d_rightDisp[i * width + k] - d) > 1)) {
			// masked or left-right inconsistent pixel -> invalid
			d_leftDisp[i * width + j] = 0;
		}
	}
}

namespace sgm {
	namespace details {

		void check_consistency(uint16_t* d_left_disp, const uint16_t* d_right_disp, const void* d_src_left, int width, int height, int depth_bits) {

			const dim3 blocks(width / 16, height / 16);
			const dim3 threads(16, 16);
			if (depth_bits == 16) {
				check_consistency_kernel<uint16_t> << < blocks, threads >> > (d_left_disp, d_right_disp, (uint16_t*)d_src_left, width, height);
			}
			else if (depth_bits == 8) {
				check_consistency_kernel<uint8_t> << < blocks, threads >> > (d_left_disp, d_right_disp, (uint8_t*)d_src_left, width, height);
			}
			
			CudaKernelCheck();	
		}

	}
}
