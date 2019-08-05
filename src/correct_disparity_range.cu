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

#include <libsgm.h>
#include "internal.h"
#include "utility.hpp"

namespace {
	__global__ void correct_disparity_range_kernel(uint16_t* d_disp, int width, int height, int pitch, bool subpixel, int min_disp) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) {
			return;
		}

		const int scale = subpixel ? sgm::StereoSGM::SUBPIXEL_SCALE : 1;
		uint16_t d = d_disp[y * pitch + x];
		if (d == sgm::INVALID_DISP) {
			d = (min_disp - 1) * scale;
		} else {
			d += min_disp * scale;
		}
		d_disp[y * pitch + x] = d;
	}
}

namespace sgm {
	namespace details {
		void correct_disparity_range(uint16_t* d_disp, int width, int height, int pitch, bool subpixel, int min_disp) {
			if (!subpixel && min_disp == 0) {
				return;
			}

			static constexpr int SIZE = 16;
			const dim3 blocks((width + SIZE - 1) / SIZE, (height + SIZE - 1) / SIZE);
			const dim3 threads(SIZE, SIZE);
			correct_disparity_range_kernel<<<blocks, threads>>>(d_disp, width, height, pitch, subpixel, min_disp);
		}
	}
}
