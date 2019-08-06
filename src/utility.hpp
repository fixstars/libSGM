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

#ifndef SGM_UTILITY_HPP
#define SGM_UTILITY_HPP

#include <cuda.h>
#include "types.hpp"

namespace sgm {

static constexpr unsigned int WARP_SIZE = 32u;
static constexpr output_type INVALID_DISP = static_cast<output_type>(-1);

namespace detail {
	template <typename T, unsigned int GROUP_SIZE, unsigned int STEP>
	struct subgroup_min_impl {
		static __device__ T call(T x, uint32_t mask){
#if CUDA_VERSION >= 9000
			x = min(x, __shfl_xor_sync(mask, x, STEP / 2, GROUP_SIZE));
#else
			x = min(x, __shfl_xor(x, STEP / 2, GROUP_SIZE));
#endif
			return subgroup_min_impl<T, GROUP_SIZE, STEP / 2>::call(x, mask);
		}
	};
	template <typename T, unsigned int GROUP_SIZE>
	struct subgroup_min_impl<T, GROUP_SIZE, 1u> {
		static __device__ T call(T x, uint32_t){
			return x;
		}
	};
	template <unsigned int GROUP_SIZE, unsigned int STEP>
	struct subgroup_and_impl {
		static __device__ bool call(bool x, uint32_t mask){
#if CUDA_VERSION >= 9000
			x &= __shfl_xor_sync(mask, x, STEP / 2, GROUP_SIZE);
#else
			x &= __shfl_xor(x, STEP / 2, GROUP_SIZE);
#endif
			return subgroup_and_impl<GROUP_SIZE, STEP / 2>::call(x, mask);
		}
	};
	template <unsigned int GROUP_SIZE>
	struct subgroup_and_impl<GROUP_SIZE, 1u> {
		static __device__ bool call(bool x, uint32_t){
			return x;
		}
	};
}

template <unsigned int GROUP_SIZE, typename T>
__device__ inline T subgroup_min(T x, uint32_t mask){
	return detail::subgroup_min_impl<T, GROUP_SIZE, GROUP_SIZE>::call(x, mask);
}

template <unsigned int GROUP_SIZE>
__device__ inline bool subgroup_and(bool x, uint32_t mask){
	return detail::subgroup_and_impl<GROUP_SIZE, GROUP_SIZE>::call(x, mask);
}

template <typename T, typename S>
__device__ inline T load_as(const S *p){
	return *reinterpret_cast<const T *>(p);
}

template <typename T, typename S>
__device__ inline void store_as(S *p, const T& x){
	*reinterpret_cast<T *>(p) = x;
}


template <typename T>
__device__ inline uint32_t pack_uint8x4(T x, T y, T z, T w){
	uchar4 uint8x4;
	uint8x4.x = static_cast<uint8_t>(x);
	uint8x4.y = static_cast<uint8_t>(y);
	uint8x4.z = static_cast<uint8_t>(z);
	uint8x4.w = static_cast<uint8_t>(w);
	return load_as<uint32_t>(&uint8x4);
}


template <unsigned int N>
__device__ inline void load_uint8_vector(uint32_t *dest, const uint8_t *ptr);

template <>
__device__ inline void load_uint8_vector<1u>(uint32_t *dest, const uint8_t *ptr){
	dest[0] = static_cast<uint32_t>(ptr[0]);
}

template <>
__device__ inline void load_uint8_vector<2u>(uint32_t *dest, const uint8_t *ptr){
	const auto uint8x2 = load_as<uchar2>(ptr);
	dest[0] = uint8x2.x; dest[1] = uint8x2.y;
}

template <>
__device__ inline void load_uint8_vector<4u>(uint32_t *dest, const uint8_t *ptr){
	const auto uint8x4 = load_as<uchar4>(ptr);
	dest[0] = uint8x4.x; dest[1] = uint8x4.y; dest[2] = uint8x4.z; dest[3] = uint8x4.w;
}

template <>
__device__ inline void load_uint8_vector<8u>(uint32_t *dest, const uint8_t *ptr){
	const auto uint32x2 = load_as<uint2>(ptr);
	load_uint8_vector<4u>(dest + 0, reinterpret_cast<const uint8_t *>(&uint32x2.x));
	load_uint8_vector<4u>(dest + 4, reinterpret_cast<const uint8_t *>(&uint32x2.y));
}

template <>
__device__ inline void load_uint8_vector<16u>(uint32_t *dest, const uint8_t *ptr){
	const auto uint32x4 = load_as<uint4>(ptr);
	load_uint8_vector<4u>(dest +  0, reinterpret_cast<const uint8_t *>(&uint32x4.x));
	load_uint8_vector<4u>(dest +  4, reinterpret_cast<const uint8_t *>(&uint32x4.y));
	load_uint8_vector<4u>(dest +  8, reinterpret_cast<const uint8_t *>(&uint32x4.z));
	load_uint8_vector<4u>(dest + 12, reinterpret_cast<const uint8_t *>(&uint32x4.w));
}


template <unsigned int N>
__device__ inline void store_uint8_vector(uint8_t *dest, const uint32_t *ptr);

template <>
__device__ inline void store_uint8_vector<1u>(uint8_t *dest, const uint32_t *ptr){
	dest[0] = static_cast<uint8_t>(ptr[0]);
}

template <>
__device__ inline void store_uint8_vector<2u>(uint8_t *dest, const uint32_t *ptr){
	uchar2 uint8x2;
	uint8x2.x = static_cast<uint8_t>(ptr[0]);
	uint8x2.y = static_cast<uint8_t>(ptr[0]);
	store_as<uchar2>(dest, uint8x2);
}

template <>
__device__ inline void store_uint8_vector<4u>(uint8_t *dest, const uint32_t *ptr){
	store_as<uint32_t>(dest, pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]));
}

template <>
__device__ inline void store_uint8_vector<8u>(uint8_t *dest, const uint32_t *ptr){
	uint2 uint32x2;
	uint32x2.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
	uint32x2.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
	store_as<uint2>(dest, uint32x2);
}

template <>
__device__ inline void store_uint8_vector<16u>(uint8_t *dest, const uint32_t *ptr){
	uint4 uint32x4;
	uint32x4.x = pack_uint8x4(ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3]);
	uint32x4.y = pack_uint8x4(ptr[ 4], ptr[ 5], ptr[ 6], ptr[ 7]);
	uint32x4.z = pack_uint8x4(ptr[ 8], ptr[ 9], ptr[10], ptr[11]);
	uint32x4.w = pack_uint8x4(ptr[12], ptr[13], ptr[14], ptr[15]);
	store_as<uint4>(dest, uint32x4);
}


template <unsigned int N>
__device__ inline void load_uint16_vector(uint32_t *dest, const uint16_t *ptr);

template <>
__device__ inline void load_uint16_vector<1u>(uint32_t *dest, const uint16_t *ptr){
	dest[0] = static_cast<uint32_t>(ptr[0]);
}

template <>
__device__ inline void load_uint16_vector<2u>(uint32_t *dest, const uint16_t *ptr){
	const auto uint16x2 = load_as<ushort2>(ptr);
	dest[0] = uint16x2.x; dest[1] = uint16x2.y;
}

template <>
__device__ inline void load_uint16_vector<4u>(uint32_t *dest, const uint16_t *ptr){
	const auto uint16x4 = load_as<ushort4>(ptr);
	dest[0] = uint16x4.x; dest[1] = uint16x4.y; dest[2] = uint16x4.z; dest[3] = uint16x4.w;
}

template <>
__device__ inline void load_uint16_vector<8u>(uint32_t *dest, const uint16_t *ptr){
	const auto uint32x4 = load_as<uint4>(ptr);
	load_uint16_vector<2u>(dest + 0, reinterpret_cast<const uint16_t *>(&uint32x4.x));
	load_uint16_vector<2u>(dest + 2, reinterpret_cast<const uint16_t *>(&uint32x4.y));
	load_uint16_vector<2u>(dest + 4, reinterpret_cast<const uint16_t *>(&uint32x4.z));
	load_uint16_vector<2u>(dest + 6, reinterpret_cast<const uint16_t *>(&uint32x4.w));
}


template <unsigned int N>
__device__ inline void store_uint16_vector(uint16_t *dest, const uint32_t *ptr);

template <>
__device__ inline void store_uint16_vector<1u>(uint16_t *dest, const uint32_t *ptr){
	dest[0] = static_cast<uint16_t>(ptr[0]);
}

template <>
__device__ inline void store_uint16_vector<2u>(uint16_t *dest, const uint32_t *ptr){
	ushort2 uint16x2;
	uint16x2.x = static_cast<uint16_t>(ptr[0]);
	uint16x2.y = static_cast<uint16_t>(ptr[1]);
	store_as<ushort2>(dest, uint16x2);
}

template <>
__device__ inline void store_uint16_vector<4u>(uint16_t *dest, const uint32_t *ptr){
	ushort4 uint16x4;
	uint16x4.x = static_cast<uint16_t>(ptr[0]);
	uint16x4.y = static_cast<uint16_t>(ptr[1]);
	uint16x4.z = static_cast<uint16_t>(ptr[2]);
	uint16x4.w = static_cast<uint16_t>(ptr[3]);
	store_as<ushort4>(dest, uint16x4);
}

template <>
__device__ inline void store_uint16_vector<8u>(uint16_t *dest, const uint32_t *ptr){
	uint4 uint32x4;
	store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.x), &ptr[0]);
	store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.y), &ptr[2]);
	store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.z), &ptr[4]);
	store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.w), &ptr[6]);
	store_as<uint4>(dest, uint32x4);
}

template <>
__device__ inline void store_uint16_vector<16u>(uint16_t *dest, const uint32_t *ptr){
	store_uint16_vector<8u>(dest + 0, ptr + 0);
	store_uint16_vector<8u>(dest + 8, ptr + 8);
}

}

#endif
