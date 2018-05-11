#ifndef SGM_OBLIQUE_PATH_AGGREGATION_HPP
#define SGM_OBLIQUE_PATH_AGGREGATION_HPP

#include "types.hpp"

namespace sgm {
namespace path_aggregation {

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_upleft2downright_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	size_t width,
	size_t height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_upright2downleft_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	size_t width,
	size_t height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_downright2upleft_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	size_t width,
	size_t height,
	unsigned int p1,
	unsigned int p2,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_downleft2upright_path(
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
