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

#if CUDA_VERSION >= 9000
#define SHFL_UP(mask, var, delta, w) __shfl_up_sync((mask), (var), (delta), (w))
#define SHFL_DOWN(mask, var, delta, w) __shfl_down_sync((mask), (var), (delta), (w))
#else
#define SHFL_UP(mask, var, delta, width) __shfl_up((var), (delta), (width))
#define SHFL_DOWN(mask, var, delta, width) __shfl_down((var), (delta), (width))
#endif

namespace sgm
{

using COST_TYPE = cost_type;

namespace cost_aggregation
{

template <typename T> __device__ inline int popcnt(T x) { return 0; }
template <> __device__ inline int popcnt(uint32_t x) { return __popc(x); }
template <> __device__ inline int popcnt(uint64_t x) { return __popcll(x); }

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
			const uint32_t prev = SHFL_UP(mask, dp[DP_BLOCK_SIZE - 1], 1, WARP_SIZE);
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
			const uint32_t next = SHFL_DOWN(mask, dp0, 1, WARP_SIZE);
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

template <typename CENSUS_T>
__device__ inline CENSUS_T load_census_with_check(const CENSUS_T* ptr, int x, int w)
{
	return x >= 0 && x < w ? __ldg(ptr + x) : 0;
}

namespace vertical
{

static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

template <typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_vertical_path_kernel(
	uint8_t *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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

	__shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
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
		CENSUS_TYPE left_value;
		if (x < width) {
			left_value = left[x + y * width];
		}
		// Load right to smem
		for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE) {
			const unsigned int i = i0 + threadIdx.x;
			if (i < RIGHT_BUFFER_SIZE) {
				const int right_x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
				const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
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
			CENSUS_TYPE right_values[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				local_costs[j] = popcnt(left_value ^ right_values[j]);
			}
			dp.update(local_costs, p1, p2, shfl_mask);
			store_uint8_vector<DP_BLOCK_SIZE>(
				&dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width],
				dp.dp);
		}
		__syncthreads();
	}
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_up2down(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_vertical_path_kernel<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_down2up(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_vertical_path_kernel<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
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

template <typename CENSUS_TYPE, int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
	uint8_t *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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

	CENSUS_TYPE right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
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

	// initialize census buffer
	{
		const int x0 = (DIRECTION > 0 ? -1 : width) - (min_disp + dp_offset);
		for (int dy = 0; dy < DP_BLOCKS_PER_THREAD; ++dy)
			for (int dx = 0; dx < DP_BLOCK_SIZE; ++dx)
				right_buffer[dy][dx] = load_census_with_check(&right[dy * feature_step], x0 - dx, width);
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
				const CENSUS_TYPE left_value = __ldg(&left[j * feature_step + x]);
				if (DIRECTION > 0) {
					const CENSUS_TYPE t = right_buffer[j][DP_BLOCK_SIZE - 1];
					for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k) {
						right_buffer[j][k] = right_buffer[j][k - 1];
					}
					right_buffer[j][0] = SHFL_UP(shfl_mask, t, 1, SUBGROUP_SIZE);
					if (lane_id == 0) {
						right_buffer[j][0] = load_census_with_check(&right[j * feature_step], x - min_disp, width);
					}
				}
				else {
					const CENSUS_TYPE t = right_buffer[j][0];
					for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k) {
						right_buffer[j][k - 1] = right_buffer[j][k];
					}
					right_buffer[j][DP_BLOCK_SIZE - 1] = SHFL_DOWN(shfl_mask, t, 1, SUBGROUP_SIZE);
					if (lane_id + 1 == SUBGROUP_SIZE) {
						right_buffer[j][DP_BLOCK_SIZE - 1] = load_census_with_check(&right[j * feature_step], x - (min_disp + dp_offset + DP_BLOCK_SIZE - 1), width);
					}
				}
				uint32_t local_costs[DP_BLOCK_SIZE];
				for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k) {
					local_costs[k] = popcnt(left_value ^ right_buffer[j][k]);
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


template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_left2right(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_horizontal_path_kernel<CENSUS_TYPE, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_right2left(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_horizontal_path_kernel<CENSUS_TYPE, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace horizontal

namespace oblique
{

static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

template <typename CENSUS_TYPE, int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_oblique_path_kernel(
	uint8_t *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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

	__shared__ CENSUS_TYPE right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
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
				const CENSUS_TYPE right_value = load_census_with_check(&right[y * width], right_x, width);
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
			const CENSUS_TYPE left_value = __ldg(&left[x + y * width]);
			CENSUS_TYPE right_values[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
				local_costs[j] = popcnt(left_value ^ right_values[j]);
			}
			dp.update(local_costs, p1, p2, shfl_mask);
			store_uint8_vector<DP_BLOCK_SIZE>(
				&dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width],
				dp.dp);
		}
		__syncthreads();
	}
}


template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_upleft2downright(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_oblique_path_kernel<CENSUS_TYPE, 1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_upright2downleft(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_oblique_path_kernel<CENSUS_TYPE, -1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_downright2upleft(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_oblique_path_kernel<CENSUS_TYPE, -1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

template <typename CENSUS_TYPE, unsigned int MAX_DISPARITY>
void aggregate_downleft2upright(
	COST_TYPE *dest,
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
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
	aggregate_oblique_path_kernel<CENSUS_TYPE, 1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, left, right, width, height, p1, p2, min_disp);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace oblique

} // namespace cost_aggregation

namespace details
{

template <typename CENSUS_TYPE, int MAX_DISPARITY>
void cost_aggregation_(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int P1, int P2, PathType path_type, int min_disp)
{
	const int width = srcL.cols;
	const int height = srcL.rows;
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	dst.create(num_paths, height * width * MAX_DISPARITY, SGM_8U);

	const CENSUS_TYPE* left = srcL.ptr<CENSUS_TYPE>();
	const CENSUS_TYPE* right = srcR.ptr<CENSUS_TYPE>();

	cudaStream_t streams[8];
	for (int i = 0; i < num_paths; i++)
		cudaStreamCreate(&streams[i]);

	cost_aggregation::vertical::aggregate_up2down<CENSUS_TYPE, MAX_DISPARITY>(
		dst.ptr<COST_TYPE>(0), left, right, width, height, P1, P2, min_disp, streams[0]);
	cost_aggregation::vertical::aggregate_down2up<CENSUS_TYPE, MAX_DISPARITY>(
		dst.ptr<COST_TYPE>(1), left, right, width, height, P1, P2, min_disp, streams[1]);
	cost_aggregation::horizontal::aggregate_left2right<CENSUS_TYPE, MAX_DISPARITY>(
		dst.ptr<COST_TYPE>(2), left, right, width, height, P1, P2, min_disp, streams[2]);
	cost_aggregation::horizontal::aggregate_right2left<CENSUS_TYPE, MAX_DISPARITY>(
		dst.ptr<COST_TYPE>(3), left, right, width, height, P1, P2, min_disp, streams[3]);

	if (path_type == PathType::SCAN_8PATH) {
		cost_aggregation::oblique::aggregate_upleft2downright<CENSUS_TYPE, MAX_DISPARITY>(
			dst.ptr<COST_TYPE>(4), left, right, width, height, P1, P2, min_disp, streams[4]);
		cost_aggregation::oblique::aggregate_upright2downleft<CENSUS_TYPE, MAX_DISPARITY>(
			dst.ptr<COST_TYPE>(5), left, right, width, height, P1, P2, min_disp, streams[5]);
		cost_aggregation::oblique::aggregate_downright2upleft<CENSUS_TYPE, MAX_DISPARITY>(
			dst.ptr<COST_TYPE>(6), left, right, width, height, P1, P2, min_disp, streams[6]);
		cost_aggregation::oblique::aggregate_downleft2upright<CENSUS_TYPE, MAX_DISPARITY>(
			dst.ptr<COST_TYPE>(7), left, right, width, height, P1, P2, min_disp, streams[7]);
	}

	for (int i = 0; i < num_paths; i++)
		cudaStreamSynchronize(streams[i]);
	for (int i = 0; i < num_paths; i++)
		cudaStreamDestroy(streams[i]);
}

void cost_aggregation(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp)
{
	SGM_ASSERT(srcL.type == srcR.type, "left and right image type must be same.");

	if (srcL.type == SGM_32U) {
		if (disp_size == 64) {
			cost_aggregation_<uint32_t, 64>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
		else if (disp_size == 128) {
			cost_aggregation_<uint32_t, 128>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
		else if (disp_size == 256) {
			cost_aggregation_<uint32_t, 256>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
	}
	else if (srcL.type == SGM_64U) {
		if (disp_size == 64) {
			cost_aggregation_<uint64_t, 64>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
		else if (disp_size == 128) {
			cost_aggregation_<uint64_t, 128>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
		else if (disp_size == 256) {
			cost_aggregation_<uint64_t, 256>(srcL, srcR, dst, P1, P2, path_type, min_disp);
		}
	}
}

} // namespace details
} // namespace sgm
