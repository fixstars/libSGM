#include "sgm.hpp"
#include "census_transform.hpp"
#include "path_aggregation.hpp"
#include "winner_takes_all.hpp"

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class SemiGlobalMatching<T, MAX_DISPARITY>::Impl {

private:
	DeviceBuffer<T> m_input_left;
	DeviceBuffer<T> m_input_right;
	CensusTransform<T> m_census_left;
	CensusTransform<T> m_census_right;
	PathAggregation<MAX_DISPARITY> m_path_aggregation;
	WinnerTakesAll<MAX_DISPARITY> m_winner_takes_all;

public:
	Impl()
		: m_input_left()
		, m_input_right()
		, m_census_left()
		, m_census_right()
		, m_path_aggregation()
		, m_winner_takes_all()
	{ }

	void enqueue(
		output_type *dest_left,
		output_type *dest_right,
		const input_type *src_left,
		const input_type *src_right,
		size_t width,
		size_t height,
		unsigned int penalty1,
		unsigned int penalty2,
		float uniqueness,
		cudaStream_t stream)
	{
		if(m_input_left.size() != width * height){
			m_input_left = DeviceBuffer<T>(width * height);
		}
		if(m_input_right.size() != width * height){
			m_input_right = DeviceBuffer<T>(width * height);
		}
		cudaMemcpyAsync(
			m_input_left.data(),
			src_left,
			sizeof(T) * width * height,
			cudaMemcpyDeviceToDevice,
			stream);
		cudaMemcpyAsync(
			m_input_right.data(),
			src_right,
			sizeof(T) * width * height,
			cudaMemcpyDeviceToDevice,
			stream);
		m_census_left.enqueue(
			m_input_left.data(), width, height, stream);
		m_census_right.enqueue(
			m_input_right.data(), width, height, stream);
		m_path_aggregation.enqueue(
			m_census_left.get_output(),
			m_census_right.get_output(),
			width, height,
			penalty1, penalty2,
			stream);
		m_winner_takes_all.enqueue(
			m_path_aggregation.get_output(),
			width, height, uniqueness,
			stream);
		cudaMemcpyAsync(
			dest_left,
			m_winner_takes_all.get_left_output(),
			sizeof(output_type) * width * height,
			cudaMemcpyDeviceToDevice,
			stream);
		cudaMemcpyAsync(
			dest_right,
			m_winner_takes_all.get_right_output(),
			sizeof(output_type) * width * height,
			cudaMemcpyDeviceToDevice,
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
	size_t width,
	size_t height,
	unsigned int penalty1,
	unsigned int penalty2,
	float uniqueness)
{
	m_impl->enqueue(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		penalty1, penalty2,
		uniqueness, 0);
	cudaStreamSynchronize(0);
}

template <typename T, size_t MAX_DISPARITY>
void SemiGlobalMatching<T, MAX_DISPARITY>::enqueue(
	output_type *dest_left,
	output_type *dest_right,
	const input_type *src_left,
	const input_type *src_right,
	size_t width,
	size_t height,
	unsigned int penalty1,
	unsigned int penalty2,
	float uniqueness,
	cudaStream_t stream)
{
	m_impl->enqueue(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		penalty1, penalty2,
		uniqueness, stream);
}


template class SemiGlobalMatching<uint8_t,   64>;
template class SemiGlobalMatching<uint8_t,  128>;
template class SemiGlobalMatching<uint16_t,  64>;
template class SemiGlobalMatching<uint16_t, 128>;

}
