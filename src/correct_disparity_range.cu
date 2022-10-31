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

#include <cuda_runtime.h>

#include "device_utility.h"
#include "host_utility.h"

namespace
{

__global__ void correct_disparity_range_kernel(uint16_t* d_disp, int width, int height, int pitch, int min_disp_scaled, int invalid_disp_scaled) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	uint16_t d = d_disp[y * pitch + x];
	if (d == sgm::INVALID_DISP) {
		d = invalid_disp_scaled;
	} else {
		d += min_disp_scaled;
	}
	d_disp[y * pitch + x] = d;
}

} // namespace

namespace sgm
{
namespace details
{

void correct_disparity_range(DeviceImage& disp, bool subpixel, int min_disp)
{
	if (!subpixel && min_disp == 0) {
		return;
	}

	const int w = disp.cols;
	const int h = disp.rows;
	constexpr int SIZE = 16;
	const dim3 blocks(divUp(w, SIZE), divUp(h, SIZE));
	const dim3 threads(SIZE, SIZE);

	const int scale = subpixel ? StereoSGM::SUBPIXEL_SCALE : 1;
	const int     min_disp_scaled =  min_disp      * scale;
	const int invalid_disp_scaled = (min_disp - 1) * scale;

	correct_disparity_range_kernel<<<blocks, threads>>>(disp.ptr<uint16_t>(), w, h, disp.step, min_disp_scaled, invalid_disp_scaled);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace details
} // namespace sgm
