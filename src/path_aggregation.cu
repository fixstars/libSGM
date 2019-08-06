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

#include "path_aggregation.hpp"
#include "vertical_path_aggregation.hpp"
#include "horizontal_path_aggregation.hpp"
#include "oblique_path_aggregation.hpp"

namespace sgm {

template <size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::PathAggregation()
	: m_cost_buffer()
{
	for(unsigned int i = 0; i < MAX_NUM_PATHS; ++i){
		cudaStreamCreate(&m_streams[i]);
		cudaEventCreate(&m_events[i]);
	}
}

template <size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::~PathAggregation(){
	for(unsigned int i = 0; i < MAX_NUM_PATHS; ++i){
		cudaStreamSynchronize(m_streams[i]);
		cudaStreamDestroy(m_streams[i]);
		cudaEventDestroy(m_events[i]);
	}
}

template <size_t MAX_DISPARITY>
void PathAggregation<MAX_DISPARITY>::enqueue(
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	PathType path_type,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const unsigned int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	const size_t buffer_size = width * height * MAX_DISPARITY * num_paths;
	if(m_cost_buffer.size() != buffer_size){
		m_cost_buffer = DeviceBuffer<cost_type>(buffer_size);
	}
	const size_t buffer_step = width * height * MAX_DISPARITY;
	cudaStreamSynchronize(stream);

	path_aggregation::enqueue_aggregate_up2down_path<MAX_DISPARITY>(
		m_cost_buffer.data() + 0 * buffer_step,
		left, right, width, height, p1, p2, min_disp, m_streams[0]);
	path_aggregation::enqueue_aggregate_down2up_path<MAX_DISPARITY>(
		m_cost_buffer.data() + 1 * buffer_step,
		left, right, width, height, p1, p2, min_disp, m_streams[1]);
	path_aggregation::enqueue_aggregate_left2right_path<MAX_DISPARITY>(
		m_cost_buffer.data() + 2 * buffer_step,
		left, right, width, height, p1, p2, min_disp, m_streams[2]);
	path_aggregation::enqueue_aggregate_right2left_path<MAX_DISPARITY>(
		m_cost_buffer.data() + 3 * buffer_step,
		left, right, width, height, p1, p2, min_disp, m_streams[3]);

	if (path_type == PathType::SCAN_8PATH) {
		path_aggregation::enqueue_aggregate_upleft2downright_path<MAX_DISPARITY>(
			m_cost_buffer.data() + 4 * buffer_step,
			left, right, width, height, p1, p2, min_disp, m_streams[4]);
		path_aggregation::enqueue_aggregate_upright2downleft_path<MAX_DISPARITY>(
			m_cost_buffer.data() + 5 * buffer_step,
			left, right, width, height, p1, p2, min_disp, m_streams[5]);
		path_aggregation::enqueue_aggregate_downright2upleft_path<MAX_DISPARITY>(
			m_cost_buffer.data() + 6 * buffer_step,
			left, right, width, height, p1, p2, min_disp, m_streams[6]);
		path_aggregation::enqueue_aggregate_downleft2upright_path<MAX_DISPARITY>(
			m_cost_buffer.data() + 7 * buffer_step,
			left, right, width, height, p1, p2, min_disp, m_streams[7]);
	}

	for(unsigned int i = 0; i < MAX_NUM_PATHS; ++i){
		cudaEventRecord(m_events[i], m_streams[i]);
		cudaStreamWaitEvent(stream, m_events[i], 0);
	}
}


template class PathAggregation< 64>;
template class PathAggregation<128>;
template class PathAggregation<256>;

}
