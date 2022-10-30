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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "libsgm.h"
#include "types.hpp"
#include "device_image.h"

#define CudaSafeCall(error) sgm::details::cuda_safe_call(error, __FILE__, __LINE__)

#define CudaKernelCheck() CudaSafeCall(cudaGetLastError())

namespace sgm {
	namespace details {

		void census_transform(const DeviceImage& src, DeviceImage& dst);

		void cost_aggregation(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
			int disp_size, int P1, int P2, PathType path_type, int min_disp);

		void winner_takes_all(const DeviceImage& src, DeviceImage& dstL, DeviceImage& dstR,
			int disp_size, float uniqueness, bool subpixel, PathType path_type);

		void median_filter(const DeviceImage& src, DeviceImage& dst);

		void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff);

		void correct_disparity_range(uint16_t* d_disp, int width, int height, int pitch, bool subpixel, int min_disp);

		void cast_16bit_8bit_array(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements);
		void cast_8bit_16bit_array(const uint8_t* arr8bits, uint16_t* arr16bits, int num_elements);

		inline void cuda_safe_call(cudaError error, const char *file, const int line)
		{
			if (error != cudaSuccess) {
				fprintf(stderr, "cuda error %s : %d %s\n", file, line, cudaGetErrorString(error));
				exit(-1);
			}
		}

	}
}
