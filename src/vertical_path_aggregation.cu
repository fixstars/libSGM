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

#include <cstdio>
#include "vertical_path_aggregation.hpp"
#include "path_aggregation_common.hpp"

namespace sgm {
namespace path_aggregation {

static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_vertical_path_kernel(
	uint8_t *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
	static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if(width == 0 || height == 0){
		return;
	}

	__shared__ feature_type right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

	const unsigned int warp_id  = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id  = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		((1u << SUBGROUP_SIZE) - 1u) << (group_id * SUBGROUP_SIZE);

	const unsigned int x =
		blockIdx.x * PATHS_PER_BLOCK +
		warp_id    * PATHS_PER_WARP +
		group_id;
	const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

	const unsigned int right0_addr =
		(right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
	const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
	const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

	for(unsigned int iter = 0; iter < height; ++iter){
		const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
		// Load left to register
		feature_type left_value;
		if(x < width){
			left_value = left[x + y * width];
		}
		// Load right to smem
		for(unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE){
			const unsigned int i = i0 + threadIdx.x;
			if(i < RIGHT_BUFFER_SIZE){
				const int x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i);
				feature_type right_value = 0;
				if(0 <= x && x < static_cast<int>(width)){
					right_value = right[x + y * width];
				}
				const unsigned int lo = i % DP_BLOCK_SIZE;
				const unsigned int hi = i / DP_BLOCK_SIZE;
				right_buffer[lo][hi] = right_value;
				if(hi > 0){
					right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
				}
			}
		}
		__syncthreads();
		// Compute
		if(x < width){
			feature_type right_values[DP_BLOCK_SIZE];
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				local_costs[j] = __popc(left_value ^ right_values[j]);
			}
			dp.update(local_costs, p1, p2, shfl_mask);
			store_uint8_vector<DP_BLOCK_SIZE>(
				&dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width],
				dp.dp);
		}
		__syncthreads();
	}
}

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_up2down_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_vertical_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2);
}

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_down2up_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2);
}


template void enqueue_aggregate_up2down_path<64u>(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template void enqueue_aggregate_up2down_path<128u>(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template void enqueue_aggregate_down2up_path<64u>(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template void enqueue_aggregate_down2up_path<128u>(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

}
}
