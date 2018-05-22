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
	__global__
		void omwc_gpu(const ushort2* src, uint16_t* dst, int width, int height) {
		int index, xinG, yinG, i;
		xinG = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + 1;
		yinG = blockDim.y * blockIdx.y + threadIdx.y;

		if (xinG < width && yinG < height) {
			index = width * yinG + xinG;
			if (xinG == width - 1 || yinG == 0 || yinG == width - 1){
				dst[index]     = src[index / 2].y;
				dst[index + 1] = src[(index + 1) / 2].x;
			}
			else if (yinG == width - 1) {
				dst[index + 1] = src[index / 2].y;
			}
			else {
				uint32_t arrL[9];
				uint32_t arrR[9];
				ushort2 arrTemp[6];

				arrTemp[0] = src[(index - width - 1) / 2];
				arrTemp[1] = src[(index - width + 1) / 2];
				arrTemp[2] = src[(index - 1) / 2];
				arrTemp[3] = src[(index + 1) / 2];
				arrTemp[4] = src[(index + width - 1) / 2];
				arrTemp[5] = src[(index + width + 1) / 2];

				arrL[0] = arrTemp[0].x;
				arrL[1] = arrTemp[0].y;
				arrR[0] = arrTemp[0].y;
				arrL[2] = arrTemp[1].x;
				arrR[1] = arrTemp[1].x;
				arrR[2] = arrTemp[1].y;
				arrL[3] = arrTemp[2].x;
				arrL[4] = arrTemp[2].y;
				arrR[3] = arrTemp[2].y;
				arrL[5] = arrTemp[3].x;
				arrR[4] = arrTemp[3].x;
				arrR[5] = arrTemp[3].y;
				arrL[6] = arrTemp[4].x;
				arrL[7] = arrTemp[4].y;
				arrR[6] = arrTemp[4].y;
				arrL[8] = arrTemp[5].x;
				arrR[7] = arrTemp[5].x;
				arrR[8] = arrTemp[5].y;

				uint32_t tempR, tempL;
#pragma unroll
				for (i = 1; i < 6; i++){
					if (arrL[0] > arrL[i]){
						tempL = arrL[0];
						arrL[0] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[0] > arrR[i]){
						tempR = arrR[0];
						arrR[0] = arrR[i];
						arrR[i] = tempR;
					}
				}
#pragma unroll
				for (i = 2; i < 6; i++){
					if (arrL[1] < arrL[i]){
						tempL = arrL[1];
						arrL[1] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[1] < arrR[i]){
						tempR = arrR[1];
						arrR[1] = arrR[i];
						arrR[i] = tempR;
					}
				}
#pragma unroll
				for (int i = 3; i < 7; i++){
					if (arrL[2]>arrL[i]){
						tempL = arrL[2];
						arrL[2] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[2]>arrR[i]){
						tempR = arrR[2];
						arrR[2] = arrR[i];
						arrR[i] = tempR;
					}
				}
#pragma unroll
				for (int i = 4; i < 7; i++){
					if (arrL[3]<arrL[i]){
						tempL = arrL[3];
						arrL[3] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[3]<arrR[i]){
						tempR = arrR[3];
						arrR[3] = arrR[i];
						arrR[i] = tempR;
					}
				}
#pragma unroll
				for (int i = 5; i < 8; i++){
					if (arrL[4]>arrL[i]){
						tempL = arrL[4];
						arrL[4] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[4]>arrR[i]){
						tempR = arrR[4];
						arrR[4] = arrR[i];
						arrR[i] = tempR;
					}
				}
#pragma unroll
				for (int i = 6; i < 8; i++){
					if (arrL[5]<arrL[i]){
						tempL = arrL[5];
						arrL[5] = arrL[i];
						arrL[i] = tempL;
					}
					if (arrR[5]<arrR[i]){
						tempR = arrR[5];
						arrR[5] = arrR[i];
						arrR[i] = tempR;
					}
				}
				dst[index]     = max(min(arrL[6], arrL[7]), min(max(arrL[6], arrL[7]), arrL[8]));
				dst[index + 1] = max(min(arrR[6], arrR[7]), min(max(arrR[6], arrR[7]), arrR[8]));
			}
		}
	}
}


namespace sgm {
	namespace details {

		void median_filter(const uint16_t* d_src, uint16_t* d_dst, int width, int height) {
			const int numthread_side_x = 64;
			const int numthread_side_y = 16;

			dim3 numBlocks((width + numthread_side_x - 1) / numthread_side_x, (height + numthread_side_y - 1) / numthread_side_y, 1);
			dim3 numThread(numthread_side_x / 2, numthread_side_y, 1);
			omwc_gpu << < numBlocks, numThread >> >(reinterpret_cast<const ushort2*>(d_src), d_dst, width, height);
		}

	}
}
