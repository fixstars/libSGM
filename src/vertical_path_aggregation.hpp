#ifndef SGM_VERTICAL_PATH_AGGREGATION_HPP
#define SGM_VERTICAL_PATH_AGGREGATION_HPP

#include "types.hpp"

namespace sgm {
namespace path_aggregation {

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_up2down_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	size_t width,
	size_t height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_down2up_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	size_t width,
	size_t height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

}
}

#endif
