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

#include "renderer.h"

__global__ void write_surface_U16_with_multiplication_kernel(cudaSurfaceObject_t dst_surface, const uint16_t* src, int width, int height, uint16_t scale) {

	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;

	uint16_t val = src[i * width + j];
	val *= scale;
	surf2Dwrite(val, dst_surface, sizeof(uint16_t) * j, i);
}

void write_surface_U16_with_multiplication(cudaSurfaceObject_t dst_surface, const uint16_t* d_src, int width, int height, uint16_t scale) {
	dim3 blocks(width / 4, height / 4);
	dim3 threads(4, 4);
	write_surface_U16_with_multiplication_kernel << < blocks, threads >> >(dst_surface, d_src, width, height, scale);
}
