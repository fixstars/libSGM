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

#define PENALTY1 20
#define PENALTY2 100

#define USE_ATOMIC
namespace {
	static const int PATHS_IN_BLOCK = 16;
	static const uint32_t v_PENALTY1 = (PENALTY1 << 16) | (PENALTY1 << 0);
	static const uint32_t v_PENALTY2 = (PENALTY2 << 16) | (PENALTY2 << 0);


	__device__ inline int min_warp(int val) {
		val = min(val, __shfl_xor(val, 16));
		val = min(val, __shfl_xor(val, 8));
		val = min(val, __shfl_xor(val, 4));
		val = min(val, __shfl_xor(val, 2));
		val = min(val, __shfl_xor(val, 1));
		return __shfl(val, 0);
	}

	template<int DISP_SIZE>
	__device__ inline void init_lcost_sh(uint16_t* sh) {
		// static_assert
		assert(0 && "invalid DISP_SIZE: must be 64 or 128");
	}

	template<> __device__ inline void init_lcost_sh<64>(uint16_t* sh) {
		*reinterpret_cast<uint32_t*>(&sh[64 * threadIdx.y + (threadIdx.x * 2)]) = 0;
	}

	template<> __device__ inline void init_lcost_sh<128>(uint16_t* sh) {
		*reinterpret_cast<uint64_t*>(&sh[128 * threadIdx.y + threadIdx.x * 4]) = 0;
	}


	template<int DISP_SIZE>
	__device__ int stereo_loop(
		int i, int j, const uint8_t* __restrict__ d_matching_cost,
		uint16_t *d_scost, int width, int height, int minCost, uint16_t *lcost_sh)
	{
		// static_assert 
		assert(0 && "invalid DISP_SIZE: must be 64 or 128");
		return 0;
	}

	template<>
	__device__ int stereo_loop<64>(
		int i, int j, const uint8_t* __restrict__ d_matching_cost,
		uint16_t *d_scost, int width, int height, int minCost, uint16_t *lcost_sh)
	{
		const int DISP_SIZE = 64;
		int idx = i * width + j;
		int k = threadIdx.x << 1; // 2 * threadIdx.x;

		// sgm_mcost_t == uint8_t
		uint16_t diff_tmp = *reinterpret_cast<const uint16_t*>(&d_matching_cost[idx * DISP_SIZE + k]);
		uint32_t diff_tmp2 = diff_tmp;
		uint32_t v_diff = __byte_perm(diff_tmp2, diff_tmp2, 0x7170); // pack( 0x00'[k+1], 0x00'[k+0])

		int shIdx = DISP_SIZE * threadIdx.y + k;

		// prev, curr, next = { {k-2, k-1}, {k+0, k+1}, {k+2, k+3} }
		uint32_t lcost_sh_curr = *reinterpret_cast<uint32_t*>(&lcost_sh[shIdx]);
		uint32_t lcost_sh_prev = __shfl_up((int)lcost_sh_curr, 1, 32);
		uint32_t lcost_sh_next = __shfl_down((int)lcost_sh_curr, 1, 32);

		// about __shlf_up / __shlf_down
		// > The source lane index will not wrap around the value of width,
		// > so effectively the lower delta lanes will be unchanged.
		// Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz3dwcKEwtM
		// if (k <= 0) {
		//   uint32_t v_cost0 = lcost_sh_curr;                                     // pack(curr.x,       curr.y)
		//   uint32_t v_cost1 = __byte_perm(lcost_sh_curr, lcost_sh_curr, 0x5432); // pack(cuur.y+p (!), curr.x+p)
		//   uint32_t v_cost2 = __byte_perm(lcost_sh_curr, lcost_sh_next, 0x5432); // pack(curr.y+p,     next.x+p)
		// }
		// if ((k+1) + 1 >= DISP_MAX) {
		//   uint32_t v_cost0 = lcost_sh_curr;                                     // pack(curr.x,   curr.y)
		//   uint32_t v_cost1 = __byte_perm(lcost_sh_prev, lcost_sh_curr, 0x5432); // pack(prev.y+p, curr.x+p)
		//   uint32_t v_cost2 = __byte_perm(lcost_sh_curr, lcost_sh_curr, 0x5432); // pack(curr.y+p, curr.x+p (!)) 
		// }
		// (!) value is not the minimum

		// cost0 access match_cost[k]
		// cost1 access match_cost[k-1]
		// cost2 access match_cost[k+1]
		uint32_t v_cost0 = lcost_sh_curr;
		uint32_t v_cost1 = __byte_perm(lcost_sh_prev, lcost_sh_curr, 0x5432); // pack(prev.y, curr.x)
		uint32_t v_cost2 = __byte_perm(lcost_sh_curr, lcost_sh_next, 0x5432); // pack(curr.y, next.x)

		v_cost1 = __vadd2(v_cost1, v_PENALTY1);
		v_cost2 = __vadd2(v_cost2, v_PENALTY1);

		uint32_t v_minCost = __byte_perm(minCost, minCost, 0x1010); // pack(minCost, minCost)

		uint32_t v_cost3 = __vadd2(v_minCost, v_PENALTY2);

		uint32_t v_tmp_a = __vminu2(v_cost0, v_cost1);
		uint32_t v_tmp_b = __vminu2(v_cost2, v_cost3);
		uint32_t cost_tmp = __vsub2(__vadd2(v_diff, __vminu2(v_tmp_a, v_tmp_b)), v_minCost);

		uint32_t* dst = reinterpret_cast<uint32_t*>(&d_scost[DISP_SIZE * idx + k]);

		// if no overflow, __vadd2(x,y) == x + y
#ifdef USE_ATOMIC
		atomicAdd(dst, cost_tmp);
#else
		*dst = *dst + cost_tmp;
#endif
		*reinterpret_cast<uint32_t*>(&lcost_sh[shIdx]) = cost_tmp;

		uint16_t cost_0 = cost_tmp >> 16;
		uint16_t cost_1 = cost_tmp & 0xffff;
		int minCostNext = min(cost_0, cost_1);
		return min_warp(minCostNext);
	}

	template<>
	__device__ int stereo_loop<128>(
		int i, int j, const uint8_t* __restrict__ d_matching_cost,
		uint16_t *d_scost, int width, int height, int minCost, uint16_t *lcost_sh) {

		const int DISP_SIZE = 128;

		int idx = i * width + j;
		int k = threadIdx.x << 2;
		int shIdx = DISP_SIZE * threadIdx.y + k;

		uint32_t diff_tmp = *reinterpret_cast<const uint32_t*>(&d_matching_cost[idx * DISP_SIZE + k]);
		const uint32_t v_zero = 0;
		uint32_t v_diff_L = __byte_perm(v_zero, diff_tmp, 0x0504); // pack( 0x00'[k+1], 0x00'[k+0])
		uint32_t v_diff_H = __byte_perm(v_zero, diff_tmp, 0x0706); // pack( 0x00'[k+3], 0x00'[k+2])

		// memory layout
		//              [            this_warp          ]
		// lcost_sh_prev lcost_sh_curr_L lcost_sh_curr_H lcost_sh_next
		// -   16bit   -

		uint32_t lcost_sh_curr_L = *reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 0]);
		uint32_t lcost_sh_curr_H = *reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 2]);

		uint32_t lcost_sh_prev = __shfl_up((int)lcost_sh_curr_H, 1, 32);
		uint32_t lcost_sh_next = __shfl_down((int)lcost_sh_curr_L, 1, 32);

		uint32_t v_cost0_L = lcost_sh_curr_L;
		uint32_t v_cost0_H = lcost_sh_curr_H;
		uint32_t v_cost1_L = __byte_perm(lcost_sh_prev, lcost_sh_curr_L, 0x5432);
		uint32_t v_cost1_H = __byte_perm(lcost_sh_curr_L, lcost_sh_curr_H, 0x5432);

		uint32_t v_cost2_L = __byte_perm(lcost_sh_curr_L, lcost_sh_curr_H, 0x5432);
		uint32_t v_cost2_H = __byte_perm(lcost_sh_curr_H, lcost_sh_next, 0x5432);

		uint32_t v_minCost = __byte_perm(minCost, minCost, 0x1010);

		uint32_t v_cost3 = __vadd2(v_minCost, v_PENALTY2);

		v_cost1_L = __vadd2(v_cost1_L, v_PENALTY1);
		v_cost2_L = __vadd2(v_cost2_L, v_PENALTY1);

		v_cost1_H = __vadd2(v_cost1_H, v_PENALTY1);
		v_cost2_H = __vadd2(v_cost2_H, v_PENALTY1);

		uint32_t v_tmp_a_L = __vminu2(v_cost0_L, v_cost1_L);
		uint32_t v_tmp_a_H = __vminu2(v_cost0_H, v_cost1_H);

		uint32_t v_tmp_b_L = __vminu2(v_cost2_L, v_cost3);
		uint32_t v_tmp_b_H = __vminu2(v_cost2_H, v_cost3);

		uint32_t cost_tmp_L = __vsub2(__vadd2(v_diff_L, __vminu2(v_tmp_a_L, v_tmp_b_L)), v_minCost);
		uint32_t cost_tmp_H = __vsub2(__vadd2(v_diff_H, __vminu2(v_tmp_a_H, v_tmp_b_H)), v_minCost);

		uint64_t* dst = reinterpret_cast<uint64_t*>(&d_scost[DISP_SIZE * idx + k]);

		uint2 cost_tmp_32x2;
		cost_tmp_32x2.x = cost_tmp_L;
		cost_tmp_32x2.y = cost_tmp_H;
		// if no overflow, __vadd2(x,y) == x + y
#ifdef USE_ATOMIC
		atomicAdd((unsigned long long int*)dst, *reinterpret_cast<unsigned long long int*>(&cost_tmp_32x2));
#else
		*dst = *reinterpret_cast<uint64_t*>(&cost_tmp_32x2);
#endif

		*reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 0]) = cost_tmp_L;
		*reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 2]) = cost_tmp_H;

		uint32_t cost_tmp = __vminu2(cost_tmp_L, cost_tmp_H);
		uint16_t cost_0 = cost_tmp >> 16;
		uint16_t cost_1 = cost_tmp & 0xffff;
		int minCostNext = min(cost_0, cost_1);
		return min_warp(minCostNext);
	}


	/* -------------------------------------------------------------------------------------------------------------------- */

	template<int DIR_ID> __device__ inline int get_idx_x(int width, int j) { return 0; }
	template<int DIR_ID> __device__ inline int get_idx_y(int height, int i) { return 0; }

	// direction 1 (rx =  1, ry =  1)
	// direction 3 (rx = -1, ry =  1)
	// direction 5 (rx = -1, ry = -1)
	// direction 7 (rx =  1, ry = -1)

	template<> __device__ inline int get_idx_x<1>(int width, int j) { return j; }
	template<> __device__ inline int get_idx_y<1>(int height, int i) { return i; }
	template<> __device__ inline int get_idx_x<3>(int width, int j) { return width - 1 - j; }
	template<> __device__ inline int get_idx_y<3>(int height, int i) { return i; }
	template<> __device__ inline int get_idx_x<5>(int width, int j) { return width - 1 - j; }
	template<> __device__ inline int get_idx_y<5>(int height, int i) { return height - 1 - i; }
	template<> __device__ inline int get_idx_x<7>(int width, int j) { return j; }
	template<> __device__ inline int get_idx_y<7>(int height, int i) { return height - 1 - i; }

	template<int DISP_SIZE, int DIR_ID>
	__global__ void compute_stereo_oblique_dir_kernel(
		const uint8_t* __restrict__ d_matching_cost, uint16_t *d_scost, int width, int height)
	{
		__shared__ uint16_t lcost_sh[DISP_SIZE * PATHS_IN_BLOCK];
		init_lcost_sh<DISP_SIZE>(lcost_sh);

		const int num_paths = width + height - 1;
		int pathIdx = blockIdx.x * PATHS_IN_BLOCK + threadIdx.y;
		if (pathIdx >= num_paths) { return; }

		int i = max(0, -(width - 1) + pathIdx);
		int j = max(0, width - 1 - pathIdx);
		int minCost = 0;
		while (i < height && j < width) {
			minCost = stereo_loop<DISP_SIZE>(get_idx_y<DIR_ID>(height, i), get_idx_x<DIR_ID>(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh);
			i++; j++;
		}
	}

	// direction 0 (rx = 1, ry = 0)
	// direction 4 (rx =-1, ry = 0)
	template<> __device__ inline int get_idx_x<0>(int width, int j) { return j; }
	template<> __device__ inline int get_idx_y<0>(int height, int i) { return i; }
	template<> __device__ inline int get_idx_x<4>(int width, int j) { return width - 1 - j; }
	template<> __device__ inline int get_idx_y<4>(int height, int i) { return i; }

	template<int DISP_SIZE, int DIR_ID>
	__global__ void compute_stereo_horizontal_dir_kernel(
		const uint8_t* __restrict__ d_matching_cost, uint16_t *d_scost, int width, int height)
	{
		__shared__ uint16_t lcost_sh[DISP_SIZE * PATHS_IN_BLOCK];
		init_lcost_sh<DISP_SIZE>(lcost_sh);
		int i = blockIdx.x * PATHS_IN_BLOCK + threadIdx.y;
		int minCost = 0;
#pragma unroll
		for (int j = 0; j < width; j++) {
			minCost = stereo_loop<DISP_SIZE>(get_idx_y<DIR_ID>(height, i), get_idx_x<DIR_ID>(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh);
		}
	}

	/* direction 2 (rx = 0, ry = 1) */
	/* direction 6 (rx = 0, ry = -1) */
	template<> __device__ inline int get_idx_x<2>(int width, int j) { return j; }
	template<> __device__ inline int get_idx_y<2>(int height, int i) { return i; }
	template<> __device__ inline int get_idx_x<6>(int width, int j) { return j; }
	template<> __device__ inline int get_idx_y<6>(int height, int i) { return height - 1 - i; }

	template<int DISP_SIZE, int DIR_ID>
	__global__ void compute_stereo_vertical_dir_kernel(
		const uint8_t* __restrict__ d_matching_cost, uint16_t *d_scost, int width, int height)
	{
		__shared__ uint16_t lcost_sh[DISP_SIZE * PATHS_IN_BLOCK];
		init_lcost_sh<DISP_SIZE>(lcost_sh);
		int j = blockIdx.x * PATHS_IN_BLOCK + threadIdx.y;
		int minCost = 0;
#pragma unroll
		for (int i = 0; i < height; i++) {
			minCost = stereo_loop<DISP_SIZE>(get_idx_y<DIR_ID>(height, i), get_idx_x<DIR_ID>(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh);
		}
	}

}

namespace sgm {
	namespace details {
		void scan_scost(const uint8_t* d_matching_cost, uint16_t* d_scost, int width, int height, int disp_size, cudaStream_t cuda_streams[]) {

			const int hor_num_paths = height;
			const dim3 hor_threads(32, PATHS_IN_BLOCK);
			const dim3 hor_blocks(hor_num_paths / PATHS_IN_BLOCK, 1);

			const int ver_num_paths = width;
			const dim3 ver_threads(32, PATHS_IN_BLOCK);
			const dim3 ver_blocks(ver_num_paths / PATHS_IN_BLOCK, 1);

			const int obl_num_paths = width + height - 1;
			const dim3 obl_threads(32, PATHS_IN_BLOCK);
			const dim3 obl_blocks((obl_num_paths + 1) / PATHS_IN_BLOCK, 1); // num_paths % PATHS_IN_BLOCK != 0

			if (disp_size == 64) {
				compute_stereo_horizontal_dir_kernel<64, 0> << <hor_blocks, hor_threads, 0, cuda_streams[0] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_horizontal_dir_kernel<64, 4> << <hor_blocks, hor_threads, 0, cuda_streams[4] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_vertical_dir_kernel<64, 2> << <ver_blocks, ver_threads, 0, cuda_streams[2] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_vertical_dir_kernel<64, 6> << <ver_blocks, ver_threads, 0, cuda_streams[6] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<64, 1> << <obl_blocks, obl_threads, 0, cuda_streams[1] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<64, 3> << <obl_blocks, obl_threads, 0, cuda_streams[3] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<64, 5> << <obl_blocks, obl_threads, 0, cuda_streams[5] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<64, 7> << <obl_blocks, obl_threads, 0, cuda_streams[7] >> >(d_matching_cost, d_scost, width, height);
			}
			else if (disp_size == 128) {
				compute_stereo_horizontal_dir_kernel<128, 0> << <hor_blocks, hor_threads, 0, cuda_streams[0] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_horizontal_dir_kernel<128, 4> << <hor_blocks, hor_threads, 0, cuda_streams[4] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_vertical_dir_kernel<128, 2> << <ver_blocks, ver_threads, 0, cuda_streams[2] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_vertical_dir_kernel<128, 6> << <ver_blocks, ver_threads, 0, cuda_streams[6] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<128, 1> << <obl_blocks, obl_threads, 0, cuda_streams[1] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<128, 3> << <obl_blocks, obl_threads, 0, cuda_streams[3] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<128, 5> << <obl_blocks, obl_threads, 0, cuda_streams[5] >> >(d_matching_cost, d_scost, width, height);
				compute_stereo_oblique_dir_kernel<128, 7> << <obl_blocks, obl_threads, 0, cuda_streams[7] >> >(d_matching_cost, d_scost, width, height);
			}
		}
	}
}

