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

static const int WTA_PIXEL_IN_BLOCK = 8;

namespace {

	__device__ inline int min_warp(int val) {
		val = min(val, __shfl_xor(val, 16));
		val = min(val, __shfl_xor(val, 8));
		val = min(val, __shfl_xor(val, 4));
		val = min(val, __shfl_xor(val, 2));
		val = min(val, __shfl_xor(val, 1));
		return __shfl(val, 0);
	}

	__global__ void winner_takes_all_kernel64(uint16_t* leftDisp, uint16_t* rightDisp, const uint16_t* __restrict__ d_cost, int width, int height)
	{
		const float uniqueness = 0.95f;
		const int DISP_SIZE = 64;
		int idx = threadIdx.x;
		int x = blockIdx.x * WTA_PIXEL_IN_BLOCK + threadIdx.y;
		int y = blockIdx.y;

		const size_t cost_offset = DISP_SIZE * (y * width + x);
		const uint16_t* current_cost = d_cost + cost_offset;
		__shared__ uint16_t tmp_costs_block[DISP_SIZE * WTA_PIXEL_IN_BLOCK];
		uint16_t* tmp_costs = &tmp_costs_block[DISP_SIZE * threadIdx.y];

		uint32_t tmp_cL1, tmp_cL2;
		uint32_t tmp_cR1, tmp_cR2;

		// right (1)
		tmp_costs[idx] = ((x + idx) >= width) ? 0xffff : *(d_cost + DISP_SIZE * (y * width + (x + idx)) + idx);
		tmp_costs[idx + 32] = ((x + (idx + 32)) >= width) ? 0xffff : *(d_cost + DISP_SIZE * (y * width + (x + idx + 32)) + idx + 32);

		tmp_cL1 = current_cost[idx];
		tmp_cL2 = current_cost[idx + 32];
		tmp_cR1 = tmp_costs[idx];
		tmp_cR2 = tmp_costs[idx + 32];

		tmp_cL1 = (tmp_cL1 << 16) + idx;
		tmp_cL2 = (tmp_cL2 << 16) + idx + 32;
		tmp_cR1 = (tmp_cR1 << 16) + idx;
		tmp_cR2 = (tmp_cR2 << 16) + idx + 32;
		//////////////////////////////////////

		int valL1 = min(tmp_cL1, tmp_cL2);
		int minTempL1 = min_warp(valL1);

		int minCostL1 = (minTempL1 >> 16);
		int minDispL1 = minTempL1 & 0xffff;
		//////////////////////////////////////

		if (idx + x >= width || idx == minDispL1) { tmp_cL1 = 0x7fffffff; }
		if (idx + 32 + x >= width || idx + 32 == minDispL1) { tmp_cL2 = 0x7fffffff; }

		int valL2 = min(tmp_cL1, tmp_cL2);
		int minTempL2 = min_warp(valL2);
		int minCostL2 = (minTempL2 >> 16);
		int minDispL2 = minTempL2 & 0xffff;
		minDispL2 = minDispL2 == 0xffff ? -1 : minDispL2;
		//////////////////////////////////////

		if (idx + x >= width) { tmp_cR1 = 0x7fffffff; }
		if (idx + 32 + x >= width) { tmp_cR2 = 0x7fffffff; }

		int valR1 = min(tmp_cR1, tmp_cR2);
		int minTempR1 = min_warp(valR1);

		int minCostR1 = (minTempR1 >> 16);
		int minDispR1 = minTempR1 & 0xffff;
		if (minDispR1 == 0xffff) { minDispR1 = -1; }

		///////////////////////////////////////////////////////////////////////////////////
		// right (2)
		tmp_costs[idx] = (idx == minDispR1 || (x + idx) >= width) ? 0xffff : tmp_costs[idx];
		tmp_costs[idx + 32] = ((idx + 32) == minDispR1 || (x + (idx + 32)) >= width) ? 0xffff : tmp_costs[idx + 32];

		tmp_cR1 = tmp_costs[idx];
		tmp_cR1 = (tmp_cR1 << 16) + idx;

		tmp_cR2 = tmp_costs[idx + 32];
		tmp_cR2 = (tmp_cR2 << 16) + idx + 32;

		if (idx + x >= width || idx == minDispR1) { tmp_cR1 = 0x7fffffff; }
		if (idx + 32 + x >= width || idx + 32 == minDispR1) { tmp_cR2 = 0x7fffffff; }

		int valR2 = min(tmp_cR1, tmp_cR2); // DS == 64
		int minTempR2 = min_warp(valR2);
		int minCostR2 = (minTempR2 >> 16);
		int minDispR2 = minTempR2 & 0xffff;
		if (minDispR2 == 0xffff) { minDispR2 = -1; }
		///////////////////////////////////////////////////////////////////////////////////

		if (idx == 0) {
			float lhv = minCostL2 * uniqueness;
			leftDisp[y * width + x] = (lhv < minCostL1 && abs(minDispL1 - minDispL2) > 1) ? 0 : minDispL1 + 1; // add "+1" 
			float rhv = minCostR2 * uniqueness;
			rightDisp[y * width + x] = (rhv < minCostR1 && abs(minDispR1 - minDispR2) > 1) ? 0 : minDispR1 + 1; // add "+1" 
		}
	}

	__global__ void winner_takes_all_kernel128(uint16_t* leftDisp, uint16_t* rightDisp, const uint16_t* __restrict__ d_cost, int width, int height)
	{
		const int DISP_SIZE = 128;
		const float uniqueness = 0.95f;

		int idx = threadIdx.x;
		int x = blockIdx.x * WTA_PIXEL_IN_BLOCK + threadIdx.y;
		int y = blockIdx.y;

		const size_t cost_offset = DISP_SIZE * (y * width + x);
		const uint16_t* current_cost = d_cost + cost_offset;
		__shared__ uint16_t tmp_costs_block[DISP_SIZE * WTA_PIXEL_IN_BLOCK];
		uint16_t* tmp_costs = &tmp_costs_block[DISP_SIZE * threadIdx.y];

		uint32_t tmp_cL1, tmp_cL2; uint32_t tmp_cL3, tmp_cL4;
		uint32_t tmp_cR1, tmp_cR2; uint32_t tmp_cR3, tmp_cR4;

		// right (1)
		const int idx_1 = idx * 4 + 0;
		const int idx_2 = idx * 4 + 1;
		const int idx_3 = idx * 4 + 2;
		const int idx_4 = idx * 4 + 3;

		// TODO optimize global memory loads
		tmp_costs[idx_1] = ((x + (idx_1)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_1)) + idx_1]; // d_cost[y][x + idx0][idx0]
		tmp_costs[idx_2] = ((x + (idx_2)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_2)) + idx_2];
		tmp_costs[idx_3] = ((x + (idx_3)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_3)) + idx_3];
		tmp_costs[idx_4] = ((x + (idx_4)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_4)) + idx_4];

		uint2 tmp_vcL1 = *reinterpret_cast<const uint2*>(&current_cost[idx_1]);
		const uint2 idx_v = make_uint2((idx_2 << 16) | idx_1, (idx_4 << 16) | idx_3);

		tmp_cR1 = tmp_costs[idx_1];
		tmp_cR2 = tmp_costs[idx_2];
		tmp_cR3 = tmp_costs[idx_3];
		tmp_cR4 = tmp_costs[idx_4];

		tmp_cL1 = __byte_perm(idx_v.x, tmp_vcL1.x, 0x5410);
		tmp_cL2 = __byte_perm(idx_v.x, tmp_vcL1.x, 0x7632);
		tmp_cL3 = __byte_perm(idx_v.y, tmp_vcL1.y, 0x5410);
		tmp_cL4 = __byte_perm(idx_v.y, tmp_vcL1.y, 0x7632);

		tmp_cR1 = (tmp_cR1 << 16) + idx_1;
		tmp_cR2 = (tmp_cR2 << 16) + idx_2;
		tmp_cR3 = (tmp_cR3 << 16) + idx_3;
		tmp_cR4 = (tmp_cR4 << 16) + idx_4;
		//////////////////////////////////////

		int valL1 = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
		int minTempL1 = min_warp(valL1);

		int minCostL1 = (minTempL1 >> 16);
		int minDispL1 = minTempL1 & 0xffff;
		//////////////////////////////////////
		if (idx_1 + x >= width || idx_1 == minDispL1) { tmp_cL1 = 0x7fffffff; }
		if (idx_2 + x >= width || idx_2 == minDispL1) { tmp_cL2 = 0x7fffffff; }
		if (idx_3 + x >= width || idx_3 == minDispL1) { tmp_cL3 = 0x7fffffff; }
		if (idx_4 + x >= width || idx_4 == minDispL1) { tmp_cL4 = 0x7fffffff; }

		int valL2 = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
		int minTempL2 = min_warp(valL2);
		int minCostL2 = (minTempL2 >> 16);
		int minDispL2 = minTempL2 & 0xffff;
		minDispL2 = minDispL2 == 0xffff ? -1 : minDispL2;
		//////////////////////////////////////

		if (idx_1 + x >= width) { tmp_cR1 = 0x7fffffff; }
		if (idx_2 + x >= width) { tmp_cR2 = 0x7fffffff; }
		if (idx_3 + x >= width) { tmp_cR3 = 0x7fffffff; }
		if (idx_4 + x >= width) { tmp_cR4 = 0x7fffffff; }

		int valR1 = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
		int minTempR1 = min_warp(valR1);

		int minCostR1 = (minTempR1 >> 16);
		int minDispR1 = minTempR1 & 0xffff;
		if (minDispR1 == 0xffff) { minDispR1 = -1; }
		///////////////////////////////////////////////////////////////////////////////////
		// right (2)
		tmp_costs[idx_1] = ((idx_1) == minDispR1 || (x + (idx_1)) >= width) ? 0xffff : tmp_costs[idx_1];
		tmp_costs[idx_2] = ((idx_2) == minDispR1 || (x + (idx_2)) >= width) ? 0xffff : tmp_costs[idx_2];
		tmp_costs[idx_3] = ((idx_3) == minDispR1 || (x + (idx_3)) >= width) ? 0xffff : tmp_costs[idx_3];
		tmp_costs[idx_4] = ((idx_4) == minDispR1 || (x + (idx_4)) >= width) ? 0xffff : tmp_costs[idx_4];

		tmp_cR1 = tmp_costs[idx_1];
		tmp_cR1 = (tmp_cR1 << 16) + idx_1;

		tmp_cR2 = tmp_costs[idx_2];
		tmp_cR2 = (tmp_cR2 << 16) + idx_2;

		tmp_cR3 = tmp_costs[idx_3];
		tmp_cR3 = (tmp_cR3 << 16) + idx_3;

		tmp_cR4 = tmp_costs[idx_4];
		tmp_cR4 = (tmp_cR4 << 16) + idx_4;

		if (idx_1 + x >= width || idx_1 == minDispR1) { tmp_cR1 = 0x7fffffff; }
		if (idx_2 + x >= width || idx_2 == minDispR1) { tmp_cR2 = 0x7fffffff; }
		if (idx_3 + x >= width || idx_3 == minDispR1) { tmp_cR3 = 0x7fffffff; }
		if (idx_4 + x >= width || idx_4 == minDispR1) { tmp_cR4 = 0x7fffffff; }

		int valR2 = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
		int minTempR2 = min_warp(valR2);
		int minCostR2 = (minTempR2 >> 16);
		int minDispR2 = minTempR2 & 0xffff;
		if (minDispR2 == 0xffff) { minDispR2 = -1; }
		///////////////////////////////////////////////////////////////////////////////////

		if (idx == 0) {
			float lhv = minCostL2 * uniqueness;
			leftDisp[y * width + x] = (lhv < minCostL1 && abs(minDispL1 - minDispL2) > 1) ? 0 : minDispL1 + 1; // add "+1" 
			float rhv = minCostR2 * uniqueness;
			rightDisp[y * width + x] = (rhv < minCostR1 && abs(minDispR1 - minDispR2) > 1) ? 0 : minDispR1 + 1; // add "+1" 
		}
	}

}



namespace sgm {
	namespace details {

		void winner_takes_all(const uint16_t* d_scost, uint16_t* d_left_disp, uint16_t* d_right_disp, int width, int height, int disp_size) {
			if (disp_size == 64) {
				dim3 blocks(width / WTA_PIXEL_IN_BLOCK, height);
				dim3 threads(32, WTA_PIXEL_IN_BLOCK);
				winner_takes_all_kernel64 << < blocks, threads >> > (d_left_disp, d_right_disp, d_scost, width, height);
			}
			else if (disp_size == 128) {
				dim3 blocks(width / WTA_PIXEL_IN_BLOCK, height);
				dim3 threads(32, WTA_PIXEL_IN_BLOCK);
				winner_takes_all_kernel128 << < blocks, threads >> > (d_left_disp, d_right_disp, d_scost, width, height);
			}
		}

	}
}