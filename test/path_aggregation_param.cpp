#include "path_aggregation_test.hpp"

INSTANTIATE_TEST_CASE_P(PathAggregationParameterTest, PathAggregationTest, testing::Combine(
	testing::Values(0, 1, 10),
	testing::Values(10, 20),
	testing::Values(120, 40)
));
