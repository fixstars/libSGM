/*Copyright 2016 Fixstars Corporation

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

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include "types.h"

namespace sgm
{

// WARP_SIZE is the true hardware wavefront width and parameterizes the whole
// aggregation/WTA design: shuffle-subgroup partitioning, the WTA per-lane data
// layout (REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE), and the device-side
// block sizing (BLOCK_SIZE = WARP_SIZE * N). On CUDA the wavefront is 32; on HIP
// it is 64 on wave64 GCN (gfx8/gfx9, e.g. gfx90a) and 32 on RDNA (gfx10/11).
//
// The DEVICE value is keyed per arch off the __GFX*__ macros, so a multi-arch
// build (e.g. gfx90a;gfx1100) compiles each device slice with its own correct
// width. The HOST must NOT drive launch geometry from a compile-time width: the
// host pass can target several arches at once and there is no single right
// answer. Launch block/grid dims are recomputed from a runtime-queried warpSize
// (host_utility.h device_warp_size()), so host and device agree on every arch.
//
// hipcc still parses the __global__ kernel bodies (and their __shared__ sizing,
// subgroup_min<WARP_SIZE>, etc.) in the host pass to emit the launch stubs, so
// WARP_SIZE needs SOME compile-time value there to parse. The fallback below is
// for that parse only; it never reaches runtime, where the device pass owns the
// real width and the host owns launch dims via device_warp_size().
#if defined(USE_HIP)
#  if defined(__HIP_DEVICE_COMPILE__)
#    if defined(__GFX8__) || defined(__GFX9__)
static constexpr unsigned int WARP_SIZE = 64u;
#    else
static constexpr unsigned int WARP_SIZE = 32u;
#    endif
#  else
static constexpr unsigned int WARP_SIZE = 64u;
#  endif
#else
static constexpr unsigned int WARP_SIZE = 32u;
#endif
static constexpr output_type INVALID_DISP = static_cast<output_type>(-1);

} // namespace sgm

#endif // !__CONSTANTS_H__
