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

#include "libsgm.h"
#include "vertical_path_aggregation.hpp"
#include "horizontal_path_aggregation.hpp"
#include "oblique_path_aggregation.hpp"

namespace sgm {

namespace details {

template <size_t MAX_DISPARITY>
void cost_aggregation_(const feature_type* left, const feature_type* right, cost_type* dst, int width, int height,
	int p1, int p2, PathType path_type, int min_disp)
{
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	const size_t buffer_step = width * height * MAX_DISPARITY;

	cudaStream_t streams[8];
	for (int i = 0; i < num_paths; i++)
		cudaStreamCreate(&streams[i]);

	path_aggregation::enqueue_aggregate_up2down_path<MAX_DISPARITY>(
		dst + 0 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[0]);
	path_aggregation::enqueue_aggregate_down2up_path<MAX_DISPARITY>(
		dst + 1 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[1]);
	path_aggregation::enqueue_aggregate_left2right_path<MAX_DISPARITY>(
		dst + 2 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[2]);
	path_aggregation::enqueue_aggregate_right2left_path<MAX_DISPARITY>(
		dst + 3 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[3]);

	if (path_type == PathType::SCAN_8PATH) {
		path_aggregation::enqueue_aggregate_upleft2downright_path<MAX_DISPARITY>(
			dst + 4 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[4]);
		path_aggregation::enqueue_aggregate_upright2downleft_path<MAX_DISPARITY>(
			dst + 5 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[5]);
		path_aggregation::enqueue_aggregate_downright2upleft_path<MAX_DISPARITY>(
			dst + 6 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[6]);
		path_aggregation::enqueue_aggregate_downleft2upright_path<MAX_DISPARITY>(
			dst + 7 * buffer_step, left, right, width, height, p1, p2, min_disp, streams[7]);
	}

	for (int i = 0; i < num_paths; i++)
		cudaStreamSynchronize(streams[i]);
	for (int i = 0; i < num_paths; i++)
		cudaStreamDestroy(streams[i]);
}

void cost_aggregation(const feature_type* srcL, const feature_type* srcR, cost_type* dst, int width, int height,
	int disp_size, int P1, int P2, PathType path_type, int min_disp)
{
	if (disp_size == 64) {
		cost_aggregation_<64u>(srcL, srcR, dst, width, height, P1, P2, path_type, min_disp);
	}
	else if (disp_size == 128) {
		cost_aggregation_<128u>(srcL, srcR, dst, width, height, P1, P2, path_type, min_disp);
	}
	else if (disp_size == 256) {
		cost_aggregation_<256u>(srcL, srcR, dst, width, height, P1, P2, path_type, min_disp);
	}
}

} // namespace details

}
