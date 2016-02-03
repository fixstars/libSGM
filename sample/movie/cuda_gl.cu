
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
