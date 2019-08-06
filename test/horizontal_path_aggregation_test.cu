#include <gtest/gtest.h>
#include "horizontal_path_aggregation.hpp"
#include "internal.h"
#include "path_aggregation_test.hpp"
#include "generator.hpp"
#include "test_utility.hpp"

#include "debug.hpp"

TEST_P(PathAggregationTest, RandomLeft2Right){
	static constexpr size_t width = 631, height = 479, disparity = 128;

	const auto left  = generate_random_sequence<sgm::feature_type>(width * height);
	const auto right = generate_random_sequence<sgm::feature_type>(width * height);
	const auto expect = path_aggregation(
		left, right, width, height, disparity, min_disp_, p1_, p2_, 1, 0);

	const auto d_left = to_device_vector(left);
	const auto d_right = to_device_vector(right);
	thrust::device_vector<sgm::cost_type> d_cost(width * height * disparity);
	sgm::path_aggregation::enqueue_aggregate_left2right_path<disparity>(
		d_cost.data().get(),
		d_left.data().get(),
		d_right.data().get(),
		width, height, p1_, p2_, min_disp_, 0);
	cudaStreamSynchronize(0);
	CudaKernelCheck();

	const auto actual = to_host_vector(d_cost);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, disparity);
}

TEST_P(PathAggregationTest, RandomRight2Left){
	static constexpr size_t width = 640, height = 480, disparity = 64;

	const auto left  = generate_random_sequence<sgm::feature_type>(width * height);
	const auto right = generate_random_sequence<sgm::feature_type>(width * height);
	const auto expect = path_aggregation(
		left, right, width, height, disparity, min_disp_, p1_, p2_, -1, 0);

	const auto d_left = to_device_vector(left);
	const auto d_right = to_device_vector(right);
	thrust::device_vector<sgm::cost_type> d_cost(width * height * disparity);
	sgm::path_aggregation::enqueue_aggregate_right2left_path<disparity>(
		d_cost.data().get(),
		d_left.data().get(),
		d_right.data().get(),
		width, height, p1_, p2_, min_disp_, 0);
	cudaStreamSynchronize(0);
	CudaKernelCheck();

	const auto actual = to_host_vector(d_cost);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, disparity);
}
