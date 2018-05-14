#ifndef SGM_WINNER_TAKES_ALL_HPP
#define SGM_WINNER_TAKES_ALL_HPP

#include "device_buffer.hpp"
#include "types.hpp"

namespace sgm {

template <size_t MAX_DISPARITY>
class WinnerTakesAll {

private:
	DeviceBuffer<output_type> m_left_buffer;
	DeviceBuffer<output_type> m_right_buffer;

public:
	WinnerTakesAll();

	const output_type *get_left_output() const {
		return m_left_buffer.data();
	}

	const output_type *get_right_output() const {
		return m_right_buffer.data();
	}

	void enqueue(
		const cost_type *src,
		size_t width,
		size_t height,
		float uniqueness,
		cudaStream_t stream);

	void enqueue(
		output_type *left,
		output_type *right,
		const cost_type *src,
		size_t width,
		size_t height,
		float uniqueness,
		cudaStream_t stream);

};

}

#endif
