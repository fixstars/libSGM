#ifndef SGM_TEST_DEBUG_HPP
#define SGM_TEST_DEBUG_HPP

#include <iostream>
#include <cstdint>

template <typename T>
inline bool debug_compare(
	const T *actual,
	const T *expect,
	size_t width,
	size_t height,
	size_t disparity)
{
	bool accept = true;
	for(size_t y = 0, i = 0; y < height; ++y){
		for(size_t x = 0; x < width; ++x){
			for(size_t k = 0; k < disparity; ++k, ++i){
				if(actual[i] != expect[i]){
					std::cerr << "(" << y << ", " << x << ", " << k << "): ";
					std::cerr << static_cast<uint64_t>(actual[i]) << " / ";
					std::cerr << static_cast<uint64_t>(expect[i]) << std::endl;
					accept = false;
				}
			}
		}
	}
	return accept;
}

#endif
