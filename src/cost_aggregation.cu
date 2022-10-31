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

namespace sgm
{
namespace cost_aggregation
{

template <unsigned int DP_BLOCK_SIZE, unsigned int SUBGROUP_SIZE>
struct DynamicProgramming
{
	static_assert(DP_BLOCK_SIZE >= 2, "DP_BLOCK_SIZE must be greater than or equal to 2");
	static_assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0, "SUBGROUP_SIZE must be a power of 2");

	uint32_t last_min;
	uint32_t dp[DP_BLOCK_SIZE];

	__device__ DynamicProgramming() : last_min(0)
	{
		for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i) { dp[i] = 0; }
	}

	__device__ void update(uint32_t *local_costs, uint32_t p1, uint32_t p2, uint32_t mask)
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
			if (lane_id != 0) { out = min(out, prev - last_min + p1); }
			out = min(out, dp[k + 1] - last_min + p1);
			lazy_out = local_min = out + local_costs[k];
		}
		for (unsigned int k = 1; k + 1 < DP_BLOCK_SIZE; ++k) {
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
			if (lane_id + 1 != SUBGROUP_SIZE) {
				out = min(out, next - last_min + p1);
			}
			dp[k - 1] = lazy_out;
			dp[k] = out + local_costs[k];
			local_min = min(local_min, dp[k]);
		}
		last_min = subgroup_min<SUBGROUP_SIZE>(local_min, mask);
	}
};

template <unsigned int SIZE>
__device__ unsigned int generate_mask()
{
	static_assert(SIZE <= 32, "SIZE must be less than or equal to 32");
	return static_cast<unsigned int>((1ull << SIZE) - 1u);
}

namespace vertical
{

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
	unsigned int p2,
	int min_disp)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
	static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if (width == 0 || height == 0) {
		return;
	}

	__shared__ feature_type right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

	const unsigned int warp_id = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

	const unsigned int x =
		blockIdx.x * PATHS_PER_BLOCK +
		warp_id * PATHS_PER_WARP +
		group_id;
	const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

	const unsigned int right0_addr =
		(right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
	const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
	const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

	for (unsigned int iter = 0; iter < height; ++iter) {
		const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
		// Load left to register
		feature_type left_value;
		if (x < width) {
			left_value = left[x + y * width];
		}
		// Load right to smem
		for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE) {
			const unsigned int i = i0 + threadIdx.x;
			if (i < RIGHT_BUFFER_SIZE) {
				const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
				feature_type right_value = 0;
				if (0 <= right_x && right_x < static_cast<int>(width)) {
					right_value = right[right_x + y * width];
				}
				const unsigned int lo = i % DP_BLOCK_SIZE;
				const unsigned int hi = i / DP_BLOCK_SIZE;
				right_buffer[lo][hi] = right_value;
				if (hi > 0) {
					right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
				}
			}
		}
		__syncthreads();
		// Compute
		if (x < width) {
			feature_type right_values[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
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
void aggregate_up2down(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_vertical_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <unsigned int MAX_DISPARITY>
void aggregate_down2up(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace vertical

namespace horizontal
{

static constexpr unsigned int DP_BLOCK_SIZE = 8u;
static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;

static constexpr unsigned int WARPS_PER_BLOCK = 4u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
	uint8_t *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int SUBGROUPS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_WARP =
		WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if (width == 0 || height == 0) {
		return;
	}

	feature_type right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp[DP_BLOCKS_PER_THREAD];

	const unsigned int warp_id = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

	const unsigned int y0 =
		PATHS_PER_BLOCK * blockIdx.x +
		PATHS_PER_WARP * warp_id +
		group_id;
	const unsigned int feature_step = SUBGROUPS_PER_WARP * width;
	const unsigned int dest_step = SUBGROUPS_PER_WARP * MAX_DISPARITY * width;
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;
	left += y0 * width;
	right += y0 * width;
	dest += y0 * MAX_DISPARITY * width;

	if (y0 >= height) {
		return;
	}

	if (DIRECTION > 0) {
		for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i) {
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				right_buffer[i][j] = 0;
			}
		}
	}
	else {
		for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i) {
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				const int x = static_cast<int>(width - (min_disp + j + dp_offset));
				if (0 <= x && x < static_cast<int>(width)) {
					right_buffer[i][j] = __ldg(&right[i * feature_step + x]);
				}
				else {
					right_buffer[i][j] = 0;
				}
			}
		}
	}

	int x0 = (DIRECTION > 0) ? 0 : static_cast<int>((width - 1) & ~(DP_BLOCK_SIZE - 1));
	for (unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE) {
		for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i) {
			const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
			if (x >= width) {
				continue;
			}
			for (unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j) {
				const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
				if (y >= height) {
					continue;
				}
				const feature_type left_value = __ldg(&left[j * feature_step + x]);
				if (DIRECTION > 0) {
					const feature_type t = right_buffer[j][DP_BLOCK_SIZE - 1];
					for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k) {
						right_buffer[j][k] = right_buffer[j][k - 1];
					}
#if CUDA_VERSION >= 9000
					right_buffer[j][0] = __shfl_up_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
#else
					right_buffer[j][0] = __shfl_up(t, 1, SUBGROUP_SIZE);
#endif
					if (lane_id == 0 && x >= min_disp) {
						right_buffer[j][0] =
							__ldg(&right[j * feature_step + x - min_disp]);
					}
				}
				else {
					const feature_type t = right_buffer[j][0];
					for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k) {
						right_buffer[j][k - 1] = right_buffer[j][k];
					}
#if CUDA_VERSION >= 9000
					right_buffer[j][DP_BLOCK_SIZE - 1] =
						__shfl_down_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
#else
					right_buffer[j][DP_BLOCK_SIZE - 1] = __shfl_down(t, 1, SUBGROUP_SIZE);
#endif
					if (lane_id + 1 == SUBGROUP_SIZE) {
						if (x >= min_disp + dp_offset + DP_BLOCK_SIZE - 1) {
							right_buffer[j][DP_BLOCK_SIZE - 1] =
								__ldg(&right[j * feature_step + x - (min_disp + dp_offset + DP_BLOCK_SIZE - 1)]);
						}
						else {
							right_buffer[j][DP_BLOCK_SIZE - 1] = 0;
						}
					}
				}
				uint32_t local_costs[DP_BLOCK_SIZE];
				for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k) {
					local_costs[k] = __popc(left_value ^ right_buffer[j][k]);
				}
				dp[j].update(local_costs, p1, p2, shfl_mask);
				store_uint8_vector<DP_BLOCK_SIZE>(
					&dest[j * dest_step + x * MAX_DISPARITY + dp_offset],
					dp[j].dp);
			}
		}
		x0 += static_cast<int>(DP_BLOCK_SIZE) * DIRECTION;
	}
}


template <unsigned int MAX_DISPARITY>
void aggregate_left2right(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_horizontal_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <unsigned int MAX_DISPARITY>
void aggregate_right2left(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	const int gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_horizontal_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace horizontal

namespace oblique
{

static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_oblique_path_kernel(
	uint8_t *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
	static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

	static_assert(X_DIRECTION == 1 || X_DIRECTION == -1, "");
	static_assert(Y_DIRECTION == 1 || Y_DIRECTION == -1, "");
	if (width == 0 || height == 0) {
		return;
	}

	__shared__ feature_type right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

	const unsigned int warp_id = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

	const int x0 =
		blockIdx.x * PATHS_PER_BLOCK +
		warp_id * PATHS_PER_WARP +
		group_id +
		(X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
	const int right_x00 =
		blockIdx.x * PATHS_PER_BLOCK +
		(X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

	const unsigned int right0_addr =
		static_cast<unsigned int>(right_x00 + PATHS_PER_BLOCK - 1 - x0) + dp_offset;
	const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
	const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

	for (unsigned int iter = 0; iter < height; ++iter) {
		const int y = static_cast<int>(Y_DIRECTION > 0 ? iter : height - 1 - iter);
		const int x = x0 + static_cast<int>(iter) * X_DIRECTION;
		const int right_x0 = right_x00 + static_cast<int>(iter) * X_DIRECTION;
		// Load right to smem
		for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE) {
			const unsigned int i = i0 + threadIdx.x;
			if (i < RIGHT_BUFFER_SIZE) {
				const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
				feature_type right_value = 0;
				if (0 <= right_x && right_x < static_cast<int>(width)) {
					right_value = right[right_x + y * width];
				}
				const unsigned int lo = i % DP_BLOCK_SIZE;
				const unsigned int hi = i / DP_BLOCK_SIZE;
				right_buffer[lo][hi] = right_value;
				if (hi > 0) {
					right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
				}
			}
		}
		__syncthreads();
		// Compute
		if (0 <= x && x < static_cast<int>(width)) {
			const feature_type left_value = __ldg(&left[x + y * width]);
			feature_type right_values[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
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
void aggregate_upleft2downright(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_oblique_path_kernel<1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <unsigned int MAX_DISPARITY>
void aggregate_upright2downleft(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_oblique_path_kernel<-1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <unsigned int MAX_DISPARITY>
void aggregate_downright2upleft(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_oblique_path_kernel<-1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <unsigned int MAX_DISPARITY>
void aggregate_downleft2upright(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	const int gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
	aggregate_oblique_path_kernel<1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace oblique

} // namespace cost_aggregation

namespace details
{

template <int MAX_DISPARITY>
void cost_aggregation_(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int P1, int P2, PathType path_type, int min_disp)
{
	const int width = srcL.cols;
	const int height = srcL.rows;
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	dst.create(num_paths, height * width * MAX_DISPARITY, SGM_8U);

	const feature_type* left = srcL.ptr<feature_type>();
	const feature_type* right = srcR.ptr<feature_type>();

	cudaStream_t streams[8];
	for (int i = 0; i < num_paths; i++)
		cudaStreamCreate(&streams[i]);

	cost_aggregation::vertical::aggregate_up2down<MAX_DISPARITY>(
		dst.ptr<cost_type>(0), left, right, width, height, P1, P2, min_disp, streams[0]);
	cost_aggregation::vertical::aggregate_down2up<MAX_DISPARITY>(
		dst.ptr<cost_type>(1), left, right, width, height, P1, P2, min_disp, streams[1]);
	cost_aggregation::horizontal::aggregate_left2right<MAX_DISPARITY>(
		dst.ptr<cost_type>(2), left, right, width, height, P1, P2, min_disp, streams[2]);
	cost_aggregation::horizontal::aggregate_right2left<MAX_DISPARITY>(
		dst.ptr<cost_type>(3), left, right, width, height, P1, P2, min_disp, streams[3]);

	if (path_type == PathType::SCAN_8PATH) {
		cost_aggregation::oblique::aggregate_upleft2downright<MAX_DISPARITY>(
			dst.ptr<cost_type>(4), left, right, width, height, P1, P2, min_disp, streams[4]);
		cost_aggregation::oblique::aggregate_upright2downleft<MAX_DISPARITY>(
			dst.ptr<cost_type>(5), left, right, width, height, P1, P2, min_disp, streams[5]);
		cost_aggregation::oblique::aggregate_downright2upleft<MAX_DISPARITY>(
			dst.ptr<cost_type>(6), left, right, width, height, P1, P2, min_disp, streams[6]);
		cost_aggregation::oblique::aggregate_downleft2upright<MAX_DISPARITY>(
			dst.ptr<cost_type>(7), left, right, width, height, P1, P2, min_disp, streams[7]);
	}

	for (int i = 0; i < num_paths; i++)
		cudaStreamSynchronize(streams[i]);
	for (int i = 0; i < num_paths; i++)
		cudaStreamDestroy(streams[i]);
}

void cost_aggregation(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp)
{
	if (disp_size == 64) {
		cost_aggregation_<64>(srcL, srcR, dst, P1, P2, path_type, min_disp);
	}
	else if (disp_size == 128) {
		cost_aggregation_<128>(srcL, srcR, dst, P1, P2, path_type, min_disp);
	}
	else if (disp_size == 256) {
		cost_aggregation_<256>(srcL, srcR, dst, P1, P2, path_type, min_disp);
	}
}

} // namespace details
} // namespace sgm
