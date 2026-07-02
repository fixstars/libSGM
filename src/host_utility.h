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

#ifndef __HOST_UTILITY_H__
#define __HOST_UTILITY_H__

#include <cstdio>
#include <stdexcept>

#include "cuda_to_hip.h"

#define CUDA_CHECK(err) \
do {\
	if (err != cudaSuccess) { \
		printf("[CUDA Error] %s (code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
	} \
} while (0)

#define SGM_ASSERT(expr, msg) \
if (!(expr)) { \
	throw std::logic_error(msg); \
} \

namespace sgm
{

static inline int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

// Runtime wavefront width of the currently selected device, cached per device.
// The device-side WARP_SIZE constant (constants.h) is fixed per arch at compile
// time via the __GFX*__ macros, but the host pass of a multi-arch build cannot
// know which arch will run, so every host launcher derives its block/grid
// geometry from this query instead of a compile-time constant. On CUDA the
// wavefront is always 32.
static inline int device_warp_size()
{
#if defined(USE_HIP)
	static thread_local int cached[64] = {0};
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	if (dev >= 0 && dev < 64 && cached[dev] != 0) {
		return cached[dev];
	}
	int warp = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, dev));
	if (warp <= 0) {
		warp = 64;
	}
	if (dev >= 0 && dev < 64) {
		cached[dev] = warp;
	}
	return warp;
#else
	return 32;
#endif
}

} // namespace sgm

#endif // !__HOST_UTILITY_H__
