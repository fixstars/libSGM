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

#include "internal.h"
#include "utility.hpp"
#include "host_utility.h"

namespace {
	template<typename SRC_T, typename DST_T>
	__global__ void check_consistency_kernel(DST_T* d_leftDisp, const DST_T* d_rightDisp, const SRC_T* d_left, int width, int height, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff) {

		const int j = blockIdx.x * blockDim.x + threadIdx.x;
		const int i = blockIdx.y * blockDim.y + threadIdx.y;

		// left-right consistency check, only on leftDisp, but could be done for rightDisp too

		SRC_T mask = d_left[i * src_pitch + j];
		DST_T org = d_leftDisp[i * dst_pitch + j];
		int d = org;
		if (subpixel) {
			d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
		}
		int k = j - d;
		if (mask == 0 || org == sgm::INVALID_DISP || (k >= 0 && k < width && LR_max_diff >= 0 && abs(d_rightDisp[i * dst_pitch + k] - d) > LR_max_diff)) {
			// masked or left-right inconsistent pixel -> invalid
			d_leftDisp[i * dst_pitch + j] = static_cast<DST_T>(sgm::INVALID_DISP);
		}
	}
}

namespace sgm {
	namespace details {

		void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff)
		{
			SGM_ASSERT(dispL.type == SGM_16U && dispR.type == SGM_16U, "");

			const int w = srcL.cols;
			const int h = srcL.rows;

			const dim3 block(16, 16);
			const dim3 grid(divUp(w, block.x), divUp(h, block.y));

			if (srcL.type == SGM_8U) {
				using SRC_T = uint8_t;
				check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
					srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
			}
			else {
				using SRC_T = uint16_t;
				check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
					srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
			}
			CUDA_CHECK(cudaGetLastError());
		}
	}
}
