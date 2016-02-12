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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CudaSafeCall(error) sgm::details::cuda_safe_call(error, __FILE__, __LINE__)

#define CudaKernelCheck() CudaSafeCall(cudaGetLastError())

namespace sgm {
	namespace details {

		void census(const void* d_src, uint64_t* d_dst, int window_width, int window_height, int width, int height, int depth_bits, cudaStream_t cuda_stream);

		void matching_cost(const uint64_t* d_left, const uint64_t* d_right, uint8_t* d_matching_cost, int width, int height, int disp_size);

		void scan_scost(const uint8_t* d_matching_cost, uint16_t* d_scost, int width, int height, int disp_size, cudaStream_t cuda_streams[]);

		void winner_takes_all(const uint16_t* d_scost, uint16_t* d_left_disp, uint16_t* d_right_disp, int width, int height, int disp_size);
		
		void median_filter(const uint16_t* d_src, uint16_t* d_dst, void* median_filter_buffer, int width, int height);

		void check_consistency(uint16_t* d_left_disp, const uint16_t* d_right_disp, const void* d_src_left, int width, int height, int depth_bits);

		void cast_16bit_8bit_array(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements);

		inline void cuda_safe_call(cudaError error, const char *file, const int line)
		{
			if (error != cudaSuccess) {
				fprintf(stderr, "cuda error %s : %d %s\n", file, line, cudaGetErrorString(error));
				exit(-1);
			}
		}

	}
}
