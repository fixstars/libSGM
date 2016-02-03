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

#include <nppi.h>

#include "internal.h"

namespace sgm {
	namespace details {

		void median_filter(const uint16_t* d_src, uint16_t* d_dst, void* median_filter_buffer, int width, int height) {
			NppiSize roi = { width, height };
			NppiSize mask = { 3, 3 };
			NppiPoint anchor = { 0, 0 };

			NppStatus status = nppiFilterMedian_16u_C1R(d_src, sizeof(Npp16u) * width, d_dst, sizeof(Npp16u) * width, roi, mask, anchor, (Npp8u*)median_filter_buffer);
			
			assert(status == 0);
		}

	}
}
