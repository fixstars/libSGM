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

#ifndef __CUDA_TO_HIP_H__
#define __CUDA_TO_HIP_H__

// Single compatibility shim for the ROCm/HIP port. On a CUDA build
// this header is a thin passthrough to <cuda_runtime.h>; on HIP it pulls in the
// HIP runtime and supplies the small set of device intrinsics that ROCm 7.x does
// not provide. The .cu/.cpp sources include this instead of <cuda_runtime.h>.
//
// The two non-mechanical concerns are documented at their definitions below:
//  - the wavefront width: device-side per-arch via __GFX*__ (constants.h); host
//    side via the runtime hipGetDeviceProperties().warpSize query in
//    host_utility.h device_warp_size(). No single compile-time host constant, so
//    a multi-arch (gfx90a;gfx1100) build stays correct on each device slice.
//  - the __v*u2/__v*u4 SIMD video intrinsics, software-emulated for HIP.

#if defined(USE_HIP)

#include <hip/hip_runtime.h>

// Alias the small CUDA runtime surface the host code uses onto HIP. The .cu
// kernel-launch syntax <<<>>> is accepted by hipcc directly; only these named
// runtime entry points and the stream/error types need mapping.
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;

#define cudaSuccess hipSuccess
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaStreamCreate hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize

#else

#include <cuda_runtime.h>

#endif // USE_HIP

#endif // !__CUDA_TO_HIP_H__
