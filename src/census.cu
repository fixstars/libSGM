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

#include <iostream>

#include "internal.h"

namespace {
	static const int HOR = 9;
	static const int VERT = 7;

	static const int threads_per_block = 16;

	template<typename SRC_T>
	__global__
		void census_kernel(int hor, int vert, SRC_T* d_source, uint64_t* d_dest, int width, int height)
	{
		const int i = threadIdx.y + blockIdx.y * blockDim.y;
		const int j = threadIdx.x + blockIdx.x * blockDim.x;
		const int offset = j + i * width;

		const int rad_h = HOR / 2;
		const int rad_v = VERT / 2;

		const int swidth = threads_per_block + HOR;
		const int sheight = threads_per_block + VERT;
		__shared__ SRC_T s_source[swidth*sheight];

		/**
		*                  *- blockDim.x
		*                 /
		*      +---------+---+ -- swidth (blockDim.x+HOR)
		*      |         |   |
		*      |    1    | 2 |
		*      |         |   |
		*      +---------+---+ -- blockDim.y
		*      |    3    | 4 |
		*      +---------+---+ -- sheight (blockDim.y+VERT)
		*/

		// 1. left-top side
		const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v;
		const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h;
		if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
			s_source[threadIdx.y*swidth + threadIdx.x] = d_source[ii*width + jj];
		}

		// 2. right side
		// 2 * blockDim.x >= swidth
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h + blockDim.x;
			if (threadIdx.x + blockDim.x < swidth && threadIdx.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[threadIdx.y*swidth + threadIdx.x + blockDim.x] = d_source[ii*width + jj];
				}
			}
		}

		// 3. bottom side
		// 2 * blockDim.y >= sheight
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v + blockDim.y;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h;
			if (threadIdx.x < swidth && threadIdx.y + blockDim.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[(threadIdx.y + blockDim.y)*swidth + threadIdx.x] = d_source[ii*width + jj];
				}
			}
		}

		// 4. right-bottom side
		// 2 * blockDim.x >= swidth && 2 * blockDim.y >= sheight
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v + blockDim.y;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h + blockDim.x;
			if (threadIdx.x + blockDim.x < swidth && threadIdx.y + blockDim.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[(threadIdx.y + blockDim.y)*swidth + threadIdx.x + blockDim.x] = d_source[ii*width + jj];
				}
			}
		}
		__syncthreads();

		// TODO can we remove this condition?
		if (rad_v <= i && i < height - rad_v && rad_h <= j && j < width - rad_h)
		{
			const int ii = threadIdx.y + rad_v;
			const int jj = threadIdx.x + rad_h;
			const int soffset = jj + ii * swidth;
			// const SRC_T c = d_source[offset];
			const SRC_T c = s_source[soffset];
			uint64_t value = 0;

			uint32_t value1 = 0, value2 = 0;

#pragma unroll
			for (int y = -rad_v; y < 0; y++) {
				for (int x = -rad_h; x <= rad_h; x++) {
					// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
					SRC_T result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
					value1 <<= 1;
					value1 += result;
				}
			}

			int y = 0;
#pragma unroll
			for (int x = -rad_h; x < 0; x++) {
				// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
				SRC_T result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value1 <<= 1;
				value1 += result;
			}

#pragma unroll
			for (int x = 1; x <= rad_h; x++) {
				// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
				SRC_T result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value2 <<= 1;
				value2 += result;
			}

#pragma unroll
			for (int y = 1; y <= rad_v; y++) {
				for (int x = -rad_h; x <= rad_h; x++) {
					// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
					SRC_T result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
					value2 <<= 1;
					value2 += result;
				}
			}

			value = (uint64_t)value2;
			value |= (uint64_t)value1 << (rad_v * (2 * rad_h + 1) + rad_h);

			d_dest[offset] = value;
		}
	}
}


namespace sgm {
namespace details {

		void census(
			const void* d_src, uint64_t* d_dst, 
			int window_width, int window_height,
			int width, int height, int depth_bits, cudaStream_t cuda_stream) {

			if (window_width != 9 || window_height != 7) {
				std::cerr << "unsupported census window, only 9x7" << std::endl;
				return;
			}

			const dim3   blocks((width + threads_per_block - 1) / threads_per_block, (height + threads_per_block - 1) / threads_per_block);
			const dim3   threads(threads_per_block, threads_per_block);

			if (depth_bits == 16) {
				census_kernel<uint16_t> << <blocks, threads, 0, cuda_stream >> > (9, 7, (uint16_t*)d_src, d_dst, width, height);
			}
			else if(depth_bits == 8) {
				census_kernel<uint8_t> << <blocks, threads, 0, cuda_stream >> > (9, 7, (uint8_t*)d_src, d_dst, width, height);
			}
		}

	}
}
