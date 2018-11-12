#include <gtest/gtest.h>
#include <utility>
#include <libsgm.h>
#include "winner_takes_all.hpp"
#include "generator.hpp"
#include "test_utility.hpp"

#include "debug.hpp"

namespace {

static constexpr size_t NUM_PATHS = 8;

thrust::host_vector<sgm::output_type> winner_takes_all_left(
	const thrust::host_vector<sgm::cost_type>& src,
	size_t width, size_t height, size_t pitch, size_t disparity,
	float uniqueness, bool subpixel)
{
	thrust::host_vector<sgm::output_type> result(pitch * height);
	for(size_t i = 0; i < height; ++i){
		for(size_t j = 0; j < width; ++j){
			std::vector<std::pair<int, int>> v;
			for(size_t k = 0; k < disparity; ++k){
				int cost_sum = 0;
				for(size_t p = 0; p < NUM_PATHS; ++p){
					cost_sum += static_cast<int>(src[
						p * disparity * width * height +
						i * disparity * width +
						j * disparity +
						k]);
				}
				v.emplace_back(cost_sum, static_cast<int>(k));
			}
			auto w = v;
			sort(v.begin(), v.end());
			if(v.size() < 2){
				result[i * pitch + j] = 0;
			}else{
				const int cost0 = v[0].first;
				const int cost1 = v[1].first;
				const int disp0 = v[0].second;
				const int disp1 = v[1].second;
				sgm::output_type dst;
				if (cost1 * uniqueness < cost0 && abs(disp0 - disp1) > 1) {
					dst = 0;
				} else {
					dst = disp0;
					if (subpixel) {
						dst <<= sgm::StereoSGM::SUBPIXEL_SHIFT;
						if (0 < disp0 && disp0 < static_cast<int>(disparity) - 1) {
							const int left = w[disp0 - 1].first;
							const int right = w[disp0 + 1].first;
							const int numer = left - right;
							const int denom = left - 2 * cost0 + right;
							dst += ((numer << sgm::StereoSGM::SUBPIXEL_SHIFT) + denom) / (2 * denom);
						}
					}
				}
				result[i * pitch + j] = dst;
			}
		}
	}
	return result;
}

thrust::host_vector<sgm::output_type> winner_takes_all_right(
	const thrust::host_vector<sgm::cost_type>& src,
	size_t width, size_t height, size_t pitch, size_t disparity,
	float uniqueness)
{
	thrust::host_vector<sgm::output_type> result(pitch * height);
	for(size_t i = 0; i < height; ++i){
		for(size_t j = 0; j < width; ++j){
			std::vector<std::pair<int, int>> v;
			for(size_t k = 0; j + k < width && k < disparity; ++k){
				int cost_sum = 0;
				for(size_t p = 0; p < NUM_PATHS; ++p){
					cost_sum += static_cast<int>(src[
						p * disparity * width * height +
						i * disparity * width +
						(j + k) * disparity +
						k]);
				}
				v.emplace_back(cost_sum, static_cast<int>(k));
			}
			sort(v.begin(), v.end());
			if(v.size() < 2){
				result[i * pitch + j] = 0;
			}else{
				const int cost0 = v[0].first;
				const int cost1 = v[1].first;
				const int disp0 = v[0].second;
				const int disp1 = v[1].second;
				result[i * pitch + j] = static_cast<sgm::output_type>(
					(cost1 * uniqueness < cost0 && abs(disp0 - disp1) > 1)
						? 0
						: disp0);
			}
		}
	}
	return result;
}

}

static void test_random_left(bool subpixel, size_t padding = 0)
{
	static constexpr size_t width = 313, height = 237, disparity = 128;
	static constexpr float uniqueness = 0.95f;
	const size_t pitch = width + padding;
	const auto input = generate_random_sequence<sgm::cost_type>(
	width * height * disparity * NUM_PATHS);
	const auto expect = winner_takes_all_left(
		input, width, height, pitch, disparity, uniqueness, subpixel);

	sgm::WinnerTakesAll<disparity> wta;
	const auto d_input = to_device_vector(input);
	wta.enqueue(d_input.data().get(), width, height, static_cast<int>(pitch), uniqueness, subpixel, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_left_output(), wta.get_left_output() + (pitch * height));
	const auto actual = to_host_vector(d_actual);

	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, 1);
}


TEST(WinnerTakesAllTest, RandomLeftNormal){
	test_random_left(false);
}

TEST(WinnerTakesAllTest, RandomLeftSubpixel){
	test_random_left(true);
}

TEST(WinnerTakesAllTest, RandomLeftNormalWithPitch){
	test_random_left(false, 27);
}

TEST(WinnerTakesAllTest, RandomLeftSubpixelWithPitch){
	test_random_left(true, 27);
}

static void test_random_right(size_t padding = 0)
{
	static constexpr size_t width = 313, height = 237, disparity = 64;
	static constexpr float uniqueness = 0.95f;
	const auto input = generate_random_sequence<sgm::cost_type>(
		width * height * disparity * NUM_PATHS);
	const auto expect = winner_takes_all_right(
		input, width, height, width, disparity, uniqueness);

	sgm::WinnerTakesAll<disparity> wta;
	const auto d_input = to_device_vector(input);
	wta.enqueue(d_input.data().get(), width, height, width, uniqueness, false, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_right_output(), wta.get_right_output() + (width * height));
	const auto actual = to_host_vector(d_actual);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, 1);
}

TEST(WinnerTakesAllTest, RandomRight){
	test_random_right();
}

TEST(WinnerTakesAllTest, RandomRightWithPitch){
	test_random_right(27);
}
