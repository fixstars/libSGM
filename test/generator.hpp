#ifndef SGM_TEST_GENERATOR_HPP
#define SGM_TEST_GENERATOR_HPP

#include <random>
#include <limits>
#include <thrust/host_vector.h>

extern std::default_random_engine g_random_engine;

template <typename T>
inline thrust::host_vector<T> generate_random_sequence(size_t n){
	std::uniform_int_distribution<T> distribution(
		std::numeric_limits<T>::min(),
		std::numeric_limits<T>::max());
	thrust::host_vector<T> seq(n);
	for(size_t i = 0; i < n; ++i){
		seq[i] = distribution(g_random_engine);
	}
	return seq;
}

#endif
