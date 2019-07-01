/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SGM_HORIZONTAL_PATH_AGGREGATION_HPP
#define SGM_HORIZONTAL_PATH_AGGREGATION_HPP

#include "types.hpp"

namespace sgm {
namespace path_aggregation {

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_left2right_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void enqueue_aggregate_right2left_path(
	cost_type *dest,
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}
}

#endif
