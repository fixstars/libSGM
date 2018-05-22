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
#include "census_transform.hpp"

namespace sgm {

namespace {

static constexpr int WINDOW_WIDTH  = 9;
static constexpr int WINDOW_HEIGHT = 7;

static constexpr int BLOCK_SIZE = 128;
static constexpr int LINES_PER_BLOCK = 16;

template <typename T>
__global__ void census_transform_kernel(
	feature_type *dest,
	const T *src,
	int width,
	int height)
{
	using pixel_type = T;
	static const int SMEM_BUFFER_SIZE = WINDOW_HEIGHT + 1;

	const int half_kw = WINDOW_WIDTH  / 2;
	const int half_kh = WINDOW_HEIGHT / 2;

	__shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int x0 = blockIdx.x * (BLOCK_SIZE - WINDOW_WIDTH + 1) - half_kw;
	const int y0 = blockIdx.y * LINES_PER_BLOCK;

	for(int i = 0; i < WINDOW_HEIGHT; ++i){
		const int x = x0 + tid, y = y0 - half_kh + i;
		pixel_type value = 0;
		if(0 <= x && x < width && 0 <= y && y < height){
			value = src[x + y * width];
		}
		smem_lines[i][tid] = value;
	}
	__syncthreads();

#pragma unroll
	for(int i = 0; i < LINES_PER_BLOCK; ++i){
		if(i + 1 < LINES_PER_BLOCK){
			// Load to smem
			const int x = x0 + tid, y = y0 + half_kh + i + 1;
			pixel_type value = 0;
			if(0 <= x && x <= width && 0 <= y && y < height){
				value = src[x + y * width];
			}
			const int smem_x = tid;
			const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
			smem_lines[smem_y][smem_x] = value;
		}

		if(half_kw <= tid && tid < BLOCK_SIZE - half_kw){
			// Compute and store
			const int x = x0 + tid, y = y0 + i;
			if(half_kw <= x && x < width - half_kw && half_kh <= y && y < height - half_kh){
				const int smem_x = tid;
				const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
				const pixel_type c = smem_lines[smem_y][smem_x];
				uint32_t lo = 0, hi = 0;
				for(int dy = -half_kh; dy < 0; ++dy){
					for(int dx = -half_kw; dx <= half_kw; ++dx){
						const int smem_y2 =
							(smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
						lo = (lo << 1) | (c > smem_lines[smem_y2][smem_x + dx]);
					}
				}
				for(int dx = -half_kw; dx < 0; ++dx){
					lo = (lo << 1) | (c > smem_lines[smem_y][smem_x + dx]);
				}
				for(int dx = 1; dx <= half_kw; ++dx){
					hi = (hi << 1) | (c > smem_lines[smem_y][smem_x + dx]);
				}
				for(int dy = 1; dy <= half_kh; ++dy){
					for(int dx = -half_kw; dx <= half_kw; ++dx){
						const int smem_y2 =
							(smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
						hi = (hi << 1) | (c > smem_lines[smem_y2][smem_x + dx]);
					}
				}
				union {
					uint64_t uint64;
					uint2 uint32x2;
				} u;
				u.uint32x2.x = hi;
				u.uint32x2.y = lo;
				dest[x + y * width] = u.uint64;
			}
		}
		__syncthreads();
	}
}

template <typename T>
void enqueue_census_transform(
	feature_type *dest,
	const T *src,
	size_t width,
	size_t height,
	cudaStream_t stream)
{
	const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
	const int height_per_block = LINES_PER_BLOCK;
	const dim3 gdim(
		(width  + width_per_block  - 1) / width_per_block,
		(height + height_per_block - 1) / height_per_block);
	const dim3 bdim(BLOCK_SIZE);
	census_transform_kernel<<<gdim, bdim, 0, stream>>>(dest, src, width, height);
}

/*
static constexpr int BLOCK_WIDTH  = 32;
static constexpr int BLOCK_HEIGHT =  8;

template <typename T>
__global__ void census_transform_kernel(
	feature_type *dest,
	const T *src,
	unsigned int width,
	unsigned int height)
{
	const int half_kw = WINDOW_WIDTH  / 2;
	const int half_kh = WINDOW_HEIGHT / 2;

	const int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH;
	const int y = threadIdx.y + blockIdx.y * BLOCK_HEIGHT;
	if(x >= width || y >= height){
		return;
	}
	if(x < half_kw || width - half_kw <= x || y < half_kh || height - half_kh <= y){
		dest[x + y * width] = 0;
		return;
	}

	const auto center = src[x + y * width];
	uint64_t result = 0;
	for(int i = -half_kh; i <= half_kh; ++i){
		for(int j = -half_kw; j <= half_kw; ++j){
			const auto value = src[(x + j) + (y + i) * width];
			result = (result << 1) | (center > value);
		}
	}
	dest[x + y * width] = result;
}

template <typename T>
void enqueue_census_transform(
	feature_type *dest,
	const T *src,
	size_t width,
	size_t height,
	cudaStream_t stream)
{
	const dim3 gdim(
		(width  + BLOCK_WIDTH  - 1) / BLOCK_WIDTH,
		(height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
	const dim3 bdim(BLOCK_WIDTH, BLOCK_HEIGHT);
	census_transform_kernel<<<gdim, bdim, 0, stream>>>(dest, src, width, height);
}
*/
}


template <typename T>
CensusTransform<T>::CensusTransform()
	: m_feature_buffer()
{ }

template <typename T>
void CensusTransform<T>::enqueue(
	const input_type *src,
	size_t width,
	size_t height,
	cudaStream_t stream)
{
	if(m_feature_buffer.size() != width * height){
		m_feature_buffer = DeviceBuffer<feature_type>(width * height);
	}
	enqueue_census_transform(
		m_feature_buffer.data(), src, width, height, stream);
}

template class CensusTransform<uint8_t>;
template class CensusTransform<uint16_t>;

}
