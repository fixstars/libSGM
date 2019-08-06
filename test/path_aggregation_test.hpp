#ifndef SGM_TEST_PATH_AGGREGATION_TEST_HPP
#define SGM_TEST_PATH_AGGREGATION_TEST_HPP

#include <limits>
#include <algorithm>
#include <thrust/host_vector.h>
#include <gtest/gtest.h>
#include "types.hpp"

#ifdef _WIN32
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif

static inline thrust::host_vector<sgm::cost_type> path_aggregation(
	const thrust::host_vector<sgm::feature_type>& left,
	const thrust::host_vector<sgm::feature_type>& right,
	int width, int height, int max_disparity, int min_disparity,
	int p1, int p2, int dx, int dy)
{
	thrust::host_vector<sgm::cost_type> result(width * height * max_disparity);
	std::vector<int> before(max_disparity);
	for(int i = (dy < 0 ? height - 1: 0); 0 <= i && i < height; i += (dy < 0 ? -1 : 1)){
		for(int j = (dx < 0 ? width - 1 : 0); 0 <= j && j < width; j += (dx < 0 ? -1 : 1)){
			const int i2 = i - dy, j2 = j - dx;
			const bool inside = (0 <= i2 && i2 < height && 0 <= j2 && j2 < width);
			for(int k = 0; k < max_disparity; ++k){
				before[k] = inside ? result[k + (j2 + i2 * width) * max_disparity] : 0;
			}
			const int min_cost = *min_element(before.begin(), before.end());
			for(int k = 0; k < max_disparity; ++k){
				const auto l = left[j + i * width];
				const auto r = (k + min_disparity > j ? 0 : right[(j - k - min_disparity) + i * width]);
				int cost = std::min(before[k] - min_cost, p2);
				if(k > 0){
					cost = std::min(cost, before[k - 1] - min_cost + p1);
				}
				if(k + 1 < max_disparity){
					cost = std::min(cost, before[k + 1] - min_cost + p1);
				}
				cost += static_cast<int>(popcnt64(l ^ r));
				result[k + (j + i * width) * max_disparity] = static_cast<uint8_t>(cost);
			}
		}
	}
	return result;
}

class PathAggregationTest : public testing::TestWithParam<std::tuple<int, int, int>>
{
public:
	int min_disp_;
	int p1_, p2_;

	virtual void SetUp(){
		std::tie(min_disp_, p1_, p2_) = GetParam();
	}
};

#endif
