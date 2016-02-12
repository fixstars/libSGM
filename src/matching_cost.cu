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
	__device__ inline uint64_t shfl_u64(uint64_t x, int lane) {
		int2 a = *reinterpret_cast<int2*>(&x);
		a.x = __shfl(a.x, lane);
		a.y = __shfl(a.y, lane);
		uint64_t* out = reinterpret_cast<uint64_t*>(&a);;
		return *out;
	}

	static const int MCOST_LINES64 = 32;
	
	__global__ void matching_cost_kernel64(
		const uint64_t* d_left, const uint64_t* d_right, uint8_t* d_matching_cost, int width, int height)
	{
		const int DISP_MAX = 64;
		__shared__ uint64_t right_buf[96 * MCOST_LINES64];
		int y = blockIdx.x * MCOST_LINES64 + threadIdx.y;
		int sh_offset = 96 * threadIdx.y;

		{ // first 64 pixel
			right_buf[sh_offset + threadIdx.x] = d_right[y * width + threadIdx.x];
			right_buf[sh_offset + threadIdx.x + 32] = d_right[y * width + threadIdx.x + 32];

			uint64_t left_warp_0 = d_left[y * width + threadIdx.x];
			uint64_t left_warp_32 = d_left[y * width + threadIdx.x + 32];
#pragma unroll
			for (int x = 0; x < 32; x++) {
				uint64_t left_val = shfl_u64(left_warp_0, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_MAX; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_MAX) + x * DISP_MAX + k;
					d_matching_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}
#pragma unroll
			for (int x = 32; x < 64; x++) {
				uint64_t left_val = shfl_u64(left_warp_32, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_MAX; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_MAX) + x * DISP_MAX + k;
					d_matching_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}
		}
		for (int x = 64; x < width; x += 32) {
			uint64_t left_warp = d_left[y * width + (x + threadIdx.x)];
			right_buf[sh_offset + threadIdx.x + 64] = d_right[y * width + (x + threadIdx.x)];
			for (int xoff = 0; xoff < 32; xoff++) {
				uint64_t left_val = shfl_u64(left_warp, xoff);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_MAX; k += 32) {
					uint64_t right_val = right_buf[sh_offset + 64 + xoff - k];
					int dst_idx = y * (width * DISP_MAX) + (x + xoff) * DISP_MAX + k;
					d_matching_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}
			right_buf[sh_offset + threadIdx.x + 0] = right_buf[sh_offset + threadIdx.x + 32];
			right_buf[sh_offset + threadIdx.x + 32] = right_buf[sh_offset + threadIdx.x + 64];
		}
	}


	static const int MCOST_LINES128 = 8;

	__global__ void matching_cost_kernel128(
		const uint64_t* d_left, const uint64_t* d_right, uint8_t* d_cost, int width, int height)
	{
		const int DISP_SIZE = 128;
		__shared__ uint64_t right_buf[(128 + 32) * MCOST_LINES128];
		int y = blockIdx.x * MCOST_LINES128 + threadIdx.y;
		int sh_offset = (128 + 32) * threadIdx.y;
		{ // first 128 pixel
#pragma unroll
			for (int t = 0; t < 128; t += 32) {
				right_buf[sh_offset + threadIdx.x + t] = d_right[y * width + threadIdx.x + t];
			}

			uint64_t left_warp_0 = d_left[y * width + threadIdx.x];
			uint64_t left_warp_32 = d_left[y * width + threadIdx.x + 32];
			uint64_t left_warp_64 = d_left[y * width + threadIdx.x + 64];
			uint64_t left_warp_96 = d_left[y * width + threadIdx.x + 96];

#pragma unroll
			for (int x = 0; x < 32; x++) {
				uint64_t left_val = shfl_u64(left_warp_0, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_SIZE; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
					d_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}

#pragma unroll
			for (int x = 32; x < 64; x++) {
				uint64_t left_val = shfl_u64(left_warp_32, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_SIZE; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
					d_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}

#pragma unroll
			for (int x = 64; x < 96; x++) {
				uint64_t left_val = shfl_u64(left_warp_64, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_SIZE; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
					d_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}

#pragma unroll
			for (int x = 96; x < 128; x++) {
				uint64_t left_val = shfl_u64(left_warp_96, x);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_SIZE; k += 32) {
					uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
					int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
					d_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}
		} // end first 128 pix



		for (int x = 128; x < width; x += 32) {
			uint64_t left_warp = d_left[y * width + (x + threadIdx.x)];
			right_buf[sh_offset + threadIdx.x + 128] = d_right[y * width + (x + threadIdx.x)];
			for (int xoff = 0; xoff < 32; xoff++) {
				uint64_t left_val = shfl_u64(left_warp, xoff);
#pragma unroll
				for (int k = threadIdx.x; k < DISP_SIZE; k += 32) {
					uint64_t right_val = right_buf[sh_offset + 128 + xoff - k];
					int dst_idx = y * (width * DISP_SIZE) + (x + xoff) * DISP_SIZE + k;
					d_cost[dst_idx] = __popcll(left_val ^ right_val);
				}
			}
			right_buf[sh_offset + threadIdx.x + 0] = right_buf[sh_offset + threadIdx.x + 32];
			right_buf[sh_offset + threadIdx.x + 32] = right_buf[sh_offset + threadIdx.x + 64];
			right_buf[sh_offset + threadIdx.x + 64] = right_buf[sh_offset + threadIdx.x + 96];
			right_buf[sh_offset + threadIdx.x + 96] = right_buf[sh_offset + threadIdx.x + 128];
		}
	}
}

namespace sgm {
	namespace details {
		void matching_cost(const uint64_t* d_left, const uint64_t* d_right, uint8_t* d_matching_cost, int width, int height, int disp_size) {
			if (disp_size == 64) {
				dim3 threads(32, MCOST_LINES64);
				dim3 blocks(height / MCOST_LINES64);
				matching_cost_kernel64 << <blocks, threads >> >(d_left, d_right, d_matching_cost, width, height);
			}
			else if (disp_size == 128) {
				dim3 threads(32, MCOST_LINES128);
				dim3 blocks(height / MCOST_LINES128);
				matching_cost_kernel128 << <blocks, threads >> >(d_left, d_right, d_matching_cost, width, height);
			}
		}
	}
}

