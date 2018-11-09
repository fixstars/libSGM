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
	int height,
	int pitch)
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
			value = src[x + y * pitch];
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
			if(0 <= x && x < width && 0 <= y && y < height){
				value = src[x + y * pitch];
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
				feature_type f = 0;
				for(int dy = -half_kh; dy < 0; ++dy){
					const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					for(int dx = -half_kw; dx <= half_kw; ++dx){
						const int smem_x1 = smem_x + dx;
						const int smem_x2 = smem_x - dx;
						const auto a = smem_lines[smem_y1][smem_x1];
						const auto b = smem_lines[smem_y2][smem_x2];
						f = (f << 1) | (a > b);
					}
				}
				for(int dx = -half_kw; dx < 0; ++dx){
					const int smem_x1 = smem_x + dx;
					const int smem_x2 = smem_x - dx;
					const auto a = smem_lines[smem_y][smem_x1];
					const auto b = smem_lines[smem_y][smem_x2];
					f = (f << 1) | (a > b);
				}
				dest[x + y * width] = f;
			}
		}
		__syncthreads();
	}
}

template <typename T>
void enqueue_census_transform(
	feature_type *dest,
	const T *src,
	int width,
	int height,
	int pitch,
	cudaStream_t stream)
{
	const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
	const int height_per_block = LINES_PER_BLOCK;
	const dim3 gdim(
		(width  + width_per_block  - 1) / width_per_block,
		(height + height_per_block - 1) / height_per_block);
	const dim3 bdim(BLOCK_SIZE);
	census_transform_kernel<<<gdim, bdim, 0, stream>>>(dest, src, width, height, pitch);
}

}


template <typename T>
CensusTransform<T>::CensusTransform()
	: m_feature_buffer()
{ }

template <typename T>
void CensusTransform<T>::enqueue(
	const input_type *src,
	int width,
	int height,
	int pitch,
	cudaStream_t stream)
{
	if(m_feature_buffer.size() != static_cast<size_t>(width * height)){
		m_feature_buffer = DeviceBuffer<feature_type>(width * height);
	}
	enqueue_census_transform(
		m_feature_buffer.data(), src, width, height, pitch, stream);
}

template class CensusTransform<uint8_t>;
template class CensusTransform<uint16_t>;

}
