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

#include "sgm.hpp"
#include "census_transform.hpp"
#include "path_aggregation.hpp"
#include "winner_takes_all.hpp"

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class SemiGlobalMatching<T, MAX_DISPARITY>::Impl {

private:
	CensusTransform<T> m_census_left;
	CensusTransform<T> m_census_right;
	PathAggregation<MAX_DISPARITY> m_path_aggregation;
	WinnerTakesAll<MAX_DISPARITY> m_winner_takes_all;

public:
	Impl()
		: m_census_left()
		, m_census_right()
		, m_path_aggregation()
		, m_winner_takes_all()
	{ }

	void enqueue(
		output_type *dest_left,
		output_type *dest_right,
		const input_type *src_left,
		const input_type *src_right,
		int width,
		int height,
		int src_pitch,
		int dst_pitch,
		const StereoSGM::Parameters& param,
		cudaStream_t stream)
	{
		m_census_left.enqueue(
			src_left, width, height, src_pitch, stream);
		m_census_right.enqueue(
			src_right, width, height, src_pitch, stream);
		m_path_aggregation.enqueue(
			m_census_left.get_output(),
			m_census_right.get_output(),
			width, height,
			param.path_type, param.P1, param.P2, param.min_disp,
			stream);
		m_winner_takes_all.enqueue(
			dest_left, dest_right,
			m_path_aggregation.get_output(),
			width, height, dst_pitch,
			param.uniqueness, param.subpixel, param.path_type,
			stream);
	}

};


template <typename T, size_t MAX_DISPARITY>
SemiGlobalMatching<T, MAX_DISPARITY>::SemiGlobalMatching()
	: m_impl(new Impl())
{ }

template <typename T, size_t MAX_DISPARITY>
SemiGlobalMatching<T, MAX_DISPARITY>::~SemiGlobalMatching() = default;


template <typename T, size_t MAX_DISPARITY>
void SemiGlobalMatching<T, MAX_DISPARITY>::execute(
	output_type *dest_left,
	output_type *dest_right,
	const input_type *src_left,
	const input_type *src_right,
	int width,
	int height,
	int src_pitch,
	int dst_pitch,
	const StereoSGM::Parameters& param)
{
	m_impl->enqueue(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		src_pitch, dst_pitch,
		param,
		0);
	cudaStreamSynchronize(0);
}

template <typename T, size_t MAX_DISPARITY>
void SemiGlobalMatching<T, MAX_DISPARITY>::enqueue(
	output_type *dest_left,
	output_type *dest_right,
	const input_type *src_left,
	const input_type *src_right,
	int width,
	int height,
	int src_pitch,
	int dst_pitch,
	const StereoSGM::Parameters& param,
	cudaStream_t stream)
{
	m_impl->enqueue(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		src_pitch, dst_pitch,
		param,
		stream);
}


template class SemiGlobalMatching<uint8_t,   64>;
template class SemiGlobalMatching<uint8_t,  128>;
template class SemiGlobalMatching<uint8_t,  256>;
template class SemiGlobalMatching<uint16_t,  64>;
template class SemiGlobalMatching<uint16_t, 128>;
template class SemiGlobalMatching<uint16_t, 256>;

}
