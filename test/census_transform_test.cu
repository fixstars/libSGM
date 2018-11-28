#include <gtest/gtest.h>
#include "census_transform.hpp"
#include "generator.hpp"
#include "test_utility.hpp"

#include "debug.hpp"

namespace {

template <typename T>
thrust::host_vector<sgm::feature_type> census_transform(
	const thrust::host_vector<T>& src, size_t width, size_t height, size_t pitch)
{
	const int hor = 9 / 2, ver = 7 / 2;
	thrust::host_vector<sgm::feature_type> result(width * height, 0);
	for(int y = ver; y < static_cast<int>(height) - ver; ++y){
		for(int x = hor; x < static_cast<int>(width) - hor; ++x){
			const auto c = src[x + y * pitch];
			sgm::feature_type value = 0;
			for(int dy = -ver; dy <= 0; ++dy){
				for(int dx = -hor; dx <= (dy == 0 ? -1 : hor); ++dx){
					const auto a = src[(x + dx) + (y + dy) * pitch];
					const auto b = src[(x - dx) + (y - dy) * pitch];
					value <<= 1;
					if(a > b){ value |= 1; }
				}
			}
			result[x + y * width] = value;
		}
	}
	return result;
}

}


TEST(CensusTransformTest, RandomU8){
	using input_type = uint8_t;
	const size_t width = 631, height = 479;
	const auto input = generate_random_sequence<input_type>(width * height);
	const auto expect = census_transform(input, width, height, width);

	sgm::CensusTransform<uint8_t> transform;
	const auto d_input = to_device_vector(input);
	transform.enqueue(d_input.data().get(), width, height, width, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::feature_type> d_actual(
		transform.get_output(), transform.get_output() + (width * height));
	const auto actual = to_host_vector(d_actual);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, 1);
}

TEST(CensusTransformTest, RandomU8WithPitch){
	using input_type = uint8_t;
	const size_t width = 631, height = 479, pitch = 640;
	const auto input = generate_random_sequence<input_type>(pitch * height);
	const auto expect = census_transform(input, width, height, pitch);

	sgm::CensusTransform<uint8_t> transform;
	const auto d_input = to_device_vector(input);
	transform.enqueue(d_input.data().get(), width, height, pitch, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::feature_type> d_actual(
		transform.get_output(), transform.get_output() + (width * height));
	const auto actual = to_host_vector(d_actual);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, 1);
}
