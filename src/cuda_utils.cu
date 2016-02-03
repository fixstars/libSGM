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

#include "internal.h"

namespace {

	__global__ void cast_16bit_8bit_array_kernel(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		arr8bits[i] = (uint8_t)arr16bits[i];
	}

}

namespace sgm {
	namespace details {

		void cast_16bit_8bit_array(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements) {
			for (int mod = 1024; mod != 0; mod >>= 1) {
				if (num_elements % mod == 0) {
					cast_16bit_8bit_array_kernel << <num_elements / mod, mod >> >(arr16bits, arr8bits, num_elements);
					break;
				}
			}
		}

	}
}
