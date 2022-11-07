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

#include "host_utility.h"

namespace
{

__global__ void cast_16bit_8bit_array_kernel(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements)
		arr8bits[i] = static_cast<uint8_t>(arr16bits[i]);
}

__global__ void cast_8bit_16bit_array_kernel(const uint8_t* arr8bits, uint16_t* arr16bits, int num_elements)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements)
		arr16bits[i] = static_cast<uint16_t>(arr8bits[i]);
}

} // namespace

namespace sgm
{
namespace details
{

void cast_16bit_to_8bit(const DeviceImage& src, DeviceImage& dst)
{
	const int w = src.cols;
	const int h = src.rows;
	dst.create(h, w, SGM_8U, src.step);

	const int num_elements = h * src.step;
	const int block = 1024;
	const int grid = divUp(num_elements, block);

	cast_16bit_8bit_array_kernel<<<grid, block>>>(src.ptr<uint16_t>(), dst.ptr<uint8_t>(), num_elements);
	CUDA_CHECK(cudaGetLastError());
}

void cast_8bit_to_16bit(const DeviceImage& src, DeviceImage& dst)
{
	const int w = src.cols;
	const int h = src.rows;
	dst.create(h, w, SGM_16U, src.step);

	const int num_elements = h * src.step;
	const int block = 1024;
	const int grid = divUp(num_elements, block);

	cast_8bit_16bit_array_kernel<<<grid, block>>>(src.ptr<uint8_t>(), dst.ptr<uint16_t>(), num_elements);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace details
} // namespace sgm
