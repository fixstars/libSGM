#include <gtest/gtest.h>
#include <utility>
#include <algorithm>
#include <libsgm.h>
#include "winner_takes_all.hpp"
#include "utility.hpp"
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
			const auto ite = std::min_element(v.begin(), v.end());
			assert(ite != v.end());
			const auto best = *ite;
			const int best_cost = best.first;
			sgm::output_type best_disp = best.second;
			sgm::output_type dst = best_disp;
			if (subpixel) {
				dst <<= sgm::StereoSGM::SUBPIXEL_SHIFT;
				if (0 < best_disp && best_disp < static_cast<int>(disparity) - 1) {
					const int left = v[best_disp - 1].first;
					const int right = v[best_disp + 1].first;
					const int numer = left - right;
					const int denom = left - 2 * best_cost + right;
					dst += ((numer << sgm::StereoSGM::SUBPIXEL_SHIFT) + denom) / (2 * denom);
				}
			}
			for (const auto& p : v) {
				const int cost = p.first;
				const int disp = p.second;
				if (cost * uniqueness < best_cost && abs(disp - best_disp) > 1) {
					dst = sgm::INVALID_DISP;
					break;
				}
			}
			result[i * pitch + j] = dst;
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
			const auto ite = std::min_element(v.begin(), v.end());
			assert(ite != v.end());
			const auto best = *ite;
			result[i * pitch + j] = best.second;
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
	wta.enqueue(d_input.data().get(), width, height, static_cast<int>(pitch), uniqueness, subpixel, sgm::PathType::SCAN_8PATH, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_left_output(), wta.get_left_output() + (pitch * height));
	const auto actual = to_host_vector(d_actual);

	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), pitch, height, 1);
}

static void test_corner1_left(bool subpixel, size_t padding = 0)
{
	static constexpr size_t width = 1, height = 1, disparity = 64;
	static constexpr float uniqueness = 0.95f;
	const size_t pitch = width + padding;
	static constexpr size_t n = width * height * disparity * NUM_PATHS;
	static constexpr size_t step = width * height * disparity;
	thrust::host_vector<sgm::cost_type> input(n);
	for (auto& v : input) {
		v = 1;
	}
	for (size_t i = 0; i < NUM_PATHS; ++i) {
		input[i * step] = 64;
	}
	const auto expect = winner_takes_all_left(
		input, width, height, pitch, disparity, uniqueness, subpixel);

	sgm::WinnerTakesAll<disparity> wta;
	const auto d_input = to_device_vector(input);
	wta.enqueue(d_input.data().get(), width, height, static_cast<int>(pitch), uniqueness, subpixel, sgm::PathType::SCAN_8PATH, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_left_output(), wta.get_left_output() + (pitch * height));
	const auto actual = to_host_vector(d_actual);

	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), pitch, height, 1);
}

static void test_corner2_left(bool subpixel, size_t padding = 0)
{
	static constexpr size_t width = 1, height = 1, disparity = 64;
	static constexpr float uniqueness = 0.95f;
	const size_t pitch = width + padding;
	static constexpr size_t n = width * height * disparity * NUM_PATHS;
	static constexpr size_t step = width * height * disparity;
	thrust::host_vector<sgm::cost_type> input(n);
	for (auto& v : input) {
		v = 64;
	}
	for (size_t i = 0; i < NUM_PATHS; ++i) {
		input[i * step + 16] = 1;
	}
	for (size_t i = 0; i < NUM_PATHS; ++i) {
		input[i * step + 32] = 1;
	}
	const auto expect = winner_takes_all_left(
		input, width, height, pitch, disparity, uniqueness, subpixel);

	sgm::WinnerTakesAll<disparity> wta;
	const auto d_input = to_device_vector(input);
	wta.enqueue(d_input.data().get(), width, height, static_cast<int>(pitch), uniqueness, subpixel, sgm::PathType::SCAN_8PATH, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_left_output(), wta.get_left_output() + (pitch * height));
	const auto actual = to_host_vector(d_actual);

	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), pitch, height, 1);
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

TEST(WinnerTakesAllTest, Corner1LeftNormal){
	test_corner1_left(false);
}

TEST(WinnerTakesAllTest, Corner1LeftSubpixel){
	test_corner1_left(true);
}

TEST(WinnerTakesAllTest, Corner2LeftNormal){
	test_corner2_left(false);
}

TEST(WinnerTakesAllTest, Corner2LeftSubpixel){
	test_corner2_left(true);
}

static void test_random_right(size_t padding = 0)
{
	static constexpr size_t width = 313, height = 237, disparity = 64;
	static constexpr float uniqueness = 0.95f;
	const size_t pitch = width + padding;
	const auto input = generate_random_sequence<sgm::cost_type>(
		width * height * disparity * NUM_PATHS);
	const auto expect = winner_takes_all_right(
		input, width, height, pitch, disparity, uniqueness);

	sgm::WinnerTakesAll<disparity> wta;
	const auto d_input = to_device_vector(input);
	wta.enqueue(d_input.data().get(), width, height, static_cast<int>(pitch), uniqueness, false, sgm::PathType::SCAN_8PATH, 0);
	cudaStreamSynchronize(0);

	const thrust::device_vector<sgm::output_type> d_actual(
		wta.get_right_output(), wta.get_right_output() + (pitch * height));
	const auto actual = to_host_vector(d_actual);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), pitch, height, 1);
}

TEST(WinnerTakesAllTest, RandomRight){
	test_random_right();
}

TEST(WinnerTakesAllTest, RandomRightWithPitch){
	test_random_right(27);
}
