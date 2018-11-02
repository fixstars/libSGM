#ifndef SGM_TEST_UTILITY_HPP
#define SGM_TEST_UTILITY_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
inline thrust::host_vector<T> to_host_vector(const thrust::device_vector<T>& src){
	thrust::host_vector<T> dest(src.size());
	cudaMemcpy(
		dest.data(),
		src.data().get(),
		sizeof(T) * src.size(),
		cudaMemcpyDeviceToHost);
	return dest;
}

template <typename T>
inline thrust::device_vector<T> to_device_vector(const thrust::host_vector<T>& src){
	thrust::device_vector<T> dest(src.size());
	cudaMemcpy(
		dest.data().get(),
		src.data(),
		sizeof(T) * src.size(),
		cudaMemcpyHostToDevice);
	return dest;
}

#endif
