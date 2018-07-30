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

#ifndef SGM_PATH_AGGREGATION_COMMON_HPP
#define SGM_PATH_AGGREGATION_COMMON_HPP

#include <cstdint>
#include <cuda.h>
#include "utility.hpp"

namespace sgm {
namespace path_aggregation {

template <
	unsigned int DP_BLOCK_SIZE,
	unsigned int SUBGROUP_SIZE>
struct DynamicProgramming {
	static_assert(
		DP_BLOCK_SIZE >= 2,
		"DP_BLOCK_SIZE must be greater than or equal to 2");
	static_assert(
		(SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0,	
		"SUBGROUP_SIZE must be a power of 2");

	uint32_t last_min;
	uint32_t dp[DP_BLOCK_SIZE];

	__device__ DynamicProgramming()
		: last_min(0)
	{
		for(unsigned int i = 0; i < DP_BLOCK_SIZE; ++i){ dp[i] = 0; }
	}

	__device__ void update(
		uint32_t *local_costs, uint32_t p1, uint32_t p2, uint32_t mask)
	{
		const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;

		const auto dp0 = dp[0];
		uint32_t lazy_out = 0, local_min = 0;
		{
			const unsigned int k = 0;
#if CUDA_VERSION >= 9000
			const uint32_t prev =
				__shfl_up_sync(mask, dp[DP_BLOCK_SIZE - 1], 1);
#else
			const uint32_t prev = __shfl_up(dp[DP_BLOCK_SIZE - 1], 1);
#endif
			uint32_t out = min(dp[k] - last_min, p2);
			if(lane_id != 0){ out = min(out, prev - last_min + p1); }
			out = min(out, dp[k + 1] - last_min + p1);
			lazy_out = local_min = out + local_costs[k];
		}
		for(unsigned int k = 1; k + 1 < DP_BLOCK_SIZE; ++k){
			uint32_t out = min(dp[k] - last_min, p2);
			out = min(out, dp[k - 1] - last_min + p1);
			out = min(out, dp[k + 1] - last_min + p1);
			dp[k - 1] = lazy_out;
			lazy_out = out + local_costs[k];
			local_min = min(local_min, lazy_out);
		}
		{
			const unsigned int k = DP_BLOCK_SIZE - 1;
#if CUDA_VERSION >= 9000
			const uint32_t next = __shfl_down_sync(mask, dp0, 1);
#else
			const uint32_t next = __shfl_down(dp0, 1);
#endif
			uint32_t out = min(dp[k] - last_min, p2);
			out = min(out, dp[k - 1] - last_min + p1);
			if(lane_id + 1 != SUBGROUP_SIZE){
				out = min(out, next - last_min + p1);
			}
			dp[k - 1] = lazy_out;
			dp[k] = out + local_costs[k];
			local_min = min(local_min, dp[k]);
		}
		last_min = subgroup_min<SUBGROUP_SIZE>(local_min, mask);
	}
};

}
}

#endif
