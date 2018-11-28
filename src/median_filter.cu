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

namespace {

	const int BLOCK_X = 16;
	const int BLOCK_Y = 16;
	const int KSIZE = 3;
	const int RADIUS = KSIZE / 2;
	const int KSIZE_SQ = KSIZE * KSIZE;

	inline int divup(int total, int grain)
	{
		return (total + grain - 1) / grain;
	}

	template <typename T>
	__device__ inline void swap(T& x, T& y)
	{
		T tmp(x);
		x = y;
		y = tmp;
	}

	// sort, min, max of 1 element
	template <typename T, int V = 1> __device__ inline void dev_sort(T& x, T& y) { if (x > y) swap(x, y); }
	template <typename T, int V = 1> __device__ inline void dev_min(T& x, T& y) { x = min(x, y); }
	template <typename T, int V = 1> __device__ inline void dev_max(T& x, T& y) { y = max(x, y); }

	// sort, min, max of 2 elements
	__device__ inline void dev_sort_2(uint32_t& x, uint32_t& y)
	{
		const uint32_t mask = __vcmpgtu2(x, y);
		const uint32_t tmp = (x ^ y) & mask;
		x ^= tmp;
		y ^= tmp;
	}
	__device__ inline void dev_min_2(uint32_t& x, uint32_t& y) { x = __vminu2(x, y); }
	__device__ inline void dev_max_2(uint32_t& x, uint32_t& y) { y = __vmaxu2(x, y); }

	template <> __device__ inline void dev_sort<uint32_t, 2>(uint32_t& x, uint32_t& y) { dev_sort_2(x, y); }
	template <> __device__ inline void dev_min<uint32_t, 2>(uint32_t& x, uint32_t& y) { dev_min_2(x, y); }
	template <> __device__ inline void dev_max<uint32_t, 2>(uint32_t& x, uint32_t& y) { dev_max_2(x, y); }

	// sort, min, max of 4 elements
	__device__ inline void dev_sort_4(uint32_t& x, uint32_t& y)
	{
		const uint32_t mask = __vcmpgtu4(x, y);
		const uint32_t tmp = (x ^ y) & mask;
		x ^= tmp;
		y ^= tmp;
	}
	__device__ inline void dev_min_4(uint32_t& x, uint32_t& y) { x = __vminu4(x, y); }
	__device__ inline void dev_max_4(uint32_t& x, uint32_t& y) { y = __vmaxu4(x, y); }

	template <> __device__ inline void dev_sort<uint32_t, 4>(uint32_t& x, uint32_t& y) { dev_sort_4(x, y); }
	template <> __device__ inline void dev_min<uint32_t, 4>(uint32_t& x, uint32_t& y) { dev_min_4(x, y); }
	template <> __device__ inline void dev_max<uint32_t, 4>(uint32_t& x, uint32_t& y) { dev_max_4(x, y); }

	template <typename T, int V = 1>
	__device__ inline void median_selection_network_9(T* buf)
	{
#define SWAP_OP(i, j) dev_sort<T, V>(buf[i], buf[j])
#define MIN_OP(i, j) dev_min<T, V>(buf[i], buf[j])
#define MAX_OP(i, j) dev_max<T, V>(buf[i], buf[j])

		SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
		SWAP_OP(1, 2); SWAP_OP(4, 5); SWAP_OP(7, 8);
		SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
		MAX_OP(0, 3); MAX_OP(3, 6);
		SWAP_OP(1, 4); MIN_OP(4, 7); MAX_OP(1, 4);
		MIN_OP(5, 8); MIN_OP(2, 5);
		SWAP_OP(2, 4); MIN_OP(4, 6); MAX_OP(2, 4);

#undef SWAP_OP
#undef MIN_OP
#undef MAX_OP
	}

	__global__ void median_kernel_3x3_8u(const uint8_t* src, uint8_t* dst, int w, int h, int p)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < RADIUS || x >= w - RADIUS || y < RADIUS || y >= h - RADIUS)
			return;

		uint8_t buf[KSIZE_SQ];
		for (int i = 0; i < KSIZE_SQ; i++)
			buf[i] = src[(y - RADIUS + i / KSIZE) * p + (x - RADIUS + i % KSIZE)];

		median_selection_network_9(buf);

		dst[y * p + x] = buf[KSIZE_SQ / 2];
	}

	__global__ void median_kernel_3x3_16u(const uint16_t* src, uint16_t* dst, int w, int h, int p)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < RADIUS || x >= w - RADIUS || y < RADIUS || y >= h - RADIUS)
			return;

		uint16_t buf[KSIZE_SQ];
		for (int i = 0; i < KSIZE_SQ; i++)
			buf[i] = src[(y - RADIUS + i / KSIZE) * p + (x - RADIUS + i % KSIZE)];

		median_selection_network_9(buf);

		dst[y * p + x] = buf[KSIZE_SQ / 2];
	}

	__global__ void median_kernel_3x3_8u_v4(const uint8_t* src, uint8_t* dst, int w, int h, int pitch)
	{
		const int x_4 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y < RADIUS || y >= h - RADIUS)
			return;

		uint32_t buf[KSIZE_SQ];
		if (x_4 >= 4 && x_4 + 7 < w)
		{
			buf[0] = *((const uint32_t*)&src[(y - 1) * pitch + x_4 - 4]);
			buf[1] = *((const uint32_t*)&src[(y - 1) * pitch + x_4 - 0]);
			buf[2] = *((const uint32_t*)&src[(y - 1) * pitch + x_4 + 4]);

			buf[3] = *((const uint32_t*)&src[(y - 0) * pitch + x_4 - 4]);
			buf[4] = *((const uint32_t*)&src[(y - 0) * pitch + x_4 - 0]);
			buf[5] = *((const uint32_t*)&src[(y - 0) * pitch + x_4 + 4]);

			buf[6] = *((const uint32_t*)&src[(y + 1) * pitch + x_4 - 4]);
			buf[7] = *((const uint32_t*)&src[(y + 1) * pitch + x_4 - 0]);
			buf[8] = *((const uint32_t*)&src[(y + 1) * pitch + x_4 + 4]);

			buf[0] = (buf[1] << 8) | (buf[0] >> 24);
			buf[2] = (buf[1] >> 8) | (buf[2] << 24);

			buf[3] = (buf[4] << 8) | (buf[3] >> 24);
			buf[5] = (buf[4] >> 8) | (buf[5] << 24);

			buf[6] = (buf[7] << 8) | (buf[6] >> 24);
			buf[8] = (buf[7] >> 8) | (buf[8] << 24);

			median_selection_network_9<uint32_t, 4>(buf);

			*((uint32_t*)&dst[y * pitch + x_4]) = buf[KSIZE_SQ / 2];
		}
		else if (x_4 == 0)
		{
			for (int x = RADIUS; x < 4; x++)
			{
				uint8_t* buf_u8 = (uint8_t*)buf;
				for (int i = 0; i < KSIZE_SQ; i++)
					buf_u8[i] = src[(y - RADIUS + i / KSIZE) * pitch + (x - RADIUS + i % KSIZE)];

				median_selection_network_9(buf_u8);

				dst[y * pitch + x] = buf_u8[KSIZE_SQ / 2];
			}
		}
		else if (x_4 < w)
		{
			for (int x = x_4; x < min(x_4 + 4, w - RADIUS); x++)
			{
				uint8_t* buf_u8 = (uint8_t*)buf;
				for (int i = 0; i < KSIZE_SQ; i++)
					buf_u8[i] = src[(y - RADIUS + i / KSIZE) * pitch + (x - RADIUS + i % KSIZE)];

				median_selection_network_9(buf_u8);

				dst[y * pitch + x] = buf_u8[KSIZE_SQ / 2];
			}
		}
	}

	__global__ void median_kernel_3x3_16u_v2(const uint16_t* src, uint16_t* dst, int w, int h, int pitch)
	{
		const int x_2 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y < RADIUS || y >= h - RADIUS)
			return;

		uint32_t buf[KSIZE_SQ];
		if (x_2 >= 2 && x_2 + 3 < w)
		{
			buf[0] = *((const uint32_t*)&src[(y - 1) * pitch + x_2 - 2]);
			buf[1] = *((const uint32_t*)&src[(y - 1) * pitch + x_2 - 0]);
			buf[2] = *((const uint32_t*)&src[(y - 1) * pitch + x_2 + 2]);

			buf[3] = *((const uint32_t*)&src[(y - 0) * pitch + x_2 - 2]);
			buf[4] = *((const uint32_t*)&src[(y - 0) * pitch + x_2 - 0]);
			buf[5] = *((const uint32_t*)&src[(y - 0) * pitch + x_2 + 2]);

			buf[6] = *((const uint32_t*)&src[(y + 1) * pitch + x_2 - 2]);
			buf[7] = *((const uint32_t*)&src[(y + 1) * pitch + x_2 - 0]);
			buf[8] = *((const uint32_t*)&src[(y + 1) * pitch + x_2 + 2]);

			buf[0] = (buf[1] << 16) | (buf[0] >> 16);
			buf[2] = (buf[1] >> 16) | (buf[2] << 16);

			buf[3] = (buf[4] << 16) | (buf[3] >> 16);
			buf[5] = (buf[4] >> 16) | (buf[5] << 16);

			buf[6] = (buf[7] << 16) | (buf[6] >> 16);
			buf[8] = (buf[7] >> 16) | (buf[8] << 16);

			median_selection_network_9<uint32_t, 2>(buf);

			*((uint32_t*)&dst[y * pitch + x_2]) = buf[KSIZE_SQ / 2];
		}
		else if (x_2 == 0)
		{
			for (int x = RADIUS; x < 2; x++)
			{
				uint8_t* buf_u8 = (uint8_t*)buf;
				for (int i = 0; i < KSIZE_SQ; i++)
					buf_u8[i] = src[(y - RADIUS + i / KSIZE) * pitch + (x - RADIUS + i % KSIZE)];

				median_selection_network_9(buf_u8);

				dst[y * pitch + x] = buf_u8[KSIZE_SQ / 2];
			}
		}
		else if (x_2 < w)
		{
			for (int x = x_2; x < min(x_2 + 2, w - RADIUS); x++)
			{
				uint8_t* buf_u8 = (uint8_t*)buf;
				for (int i = 0; i < KSIZE_SQ; i++)
					buf_u8[i] = src[(y - RADIUS + i / KSIZE) * pitch + (x - RADIUS + i % KSIZE)];

				median_selection_network_9(buf_u8);

				dst[y * pitch + x] = buf_u8[KSIZE_SQ / 2];
			}
		}
	}
}

namespace sgm {
	namespace details {

		void median_filter(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int pitch) {

			if (pitch % 4 == 0) {
				const dim3 block(BLOCK_X, BLOCK_Y);
				const dim3 grid(divup(width / 4, block.x), divup(height, block.y));
				median_kernel_3x3_8u_v4<<<grid, block>>>(d_src, d_dst, width, height, pitch);
			}
			else {
				const dim3 block(BLOCK_X, BLOCK_Y);
				const dim3 grid(divup(width, block.x), divup(height, block.y));
				median_kernel_3x3_8u<<<grid, block>>>(d_src, d_dst, width, height, pitch);
			}

			CudaSafeCall(cudaGetLastError());
		}

		void median_filter(const uint16_t* d_src, uint16_t* d_dst, int width, int height, int pitch) {
			
			if (pitch % 2 == 0) {
				const dim3 block(BLOCK_X, BLOCK_Y);
				const dim3 grid(divup(width / 2, block.x), divup(height, block.y));
				median_kernel_3x3_16u_v2<<<grid, block>>>(d_src, d_dst, width, height, pitch);
			}
			else {
				const dim3 block(BLOCK_X, BLOCK_Y);
				const dim3 grid(divup(width, block.x), divup(height, block.y));
				median_kernel_3x3_16u<<<grid, block>>>(d_src, d_dst, width, height, pitch);
			}

			CudaSafeCall(cudaGetLastError());
		}

	}
}
