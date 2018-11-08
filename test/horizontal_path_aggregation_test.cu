#include <gtest/gtest.h>
#include "horizontal_path_aggregation.hpp"
#include "path_aggregation_test.hpp"
#include "generator.hpp"
#include "test_utility.hpp"

#include "debug.hpp"

TEST(HorizontalPathAggregationTest, RandomLeft2Right){
	static constexpr size_t width = 631, height = 479, disparity = 128;
	static constexpr unsigned int p1 = 20, p2 = 100;
	const auto left  = generate_random_sequence<sgm::feature_type>(width * height);
	const auto right = generate_random_sequence<sgm::feature_type>(width * height);
	const auto expect = path_aggregation(
		left, right, width, height, disparity, p1, p2, 1, 0);

	const auto d_left = to_device_vector(left);
	const auto d_right = to_device_vector(right);
	thrust::device_vector<sgm::cost_type> d_cost(width * height * disparity);
	sgm::path_aggregation::enqueue_aggregate_left2right_path<disparity>(
		d_cost.data().get(),
		d_left.data().get(),
		d_right.data().get(),
		width, height, p1, p2, 0);
	cudaStreamSynchronize(0);

	const auto actual = to_host_vector(d_cost);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, disparity);
}

TEST(HorizontalPathAggregationTest, RandomRight2Left){
	static constexpr size_t width = 640, height = 480, disparity = 64;
	static constexpr unsigned int p1 = 20, p2 = 40;
	const auto left  = generate_random_sequence<sgm::feature_type>(width * height);
	const auto right = generate_random_sequence<sgm::feature_type>(width * height);
	const auto expect = path_aggregation(
		left, right, width, height, disparity, p1, p2, -1, 0);

	const auto d_left = to_device_vector(left);
	const auto d_right = to_device_vector(right);
	thrust::device_vector<sgm::cost_type> d_cost(width * height * disparity);
	sgm::path_aggregation::enqueue_aggregate_right2left_path<disparity>(
		d_cost.data().get(),
		d_left.data().get(),
		d_right.data().get(),
		width, height, p1, p2, 0);
	cudaStreamSynchronize(0);

	const auto actual = to_host_vector(d_cost);
	EXPECT_EQ(actual, expect);
	debug_compare(actual.data(), expect.data(), width, height, disparity);
}
