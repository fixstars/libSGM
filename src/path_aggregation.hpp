#ifndef SGM_PATH_AGGREGATION_HPP
#define SGM_PATH_AGGREGATION_HPP

#include "device_buffer.hpp"
#include "types.hpp"

namespace sgm {

template <size_t MAX_DISPARITY>
class PathAggregation {

private:
	static const unsigned int NUM_PATHS = 8;

	DeviceBuffer<cost_type> m_cost_buffer;
	cudaStream_t m_streams[NUM_PATHS];
	cudaEvent_t m_events[NUM_PATHS];
	
public:
	PathAggregation();
	~PathAggregation();

	const cost_type *get_output() const {
		return m_cost_buffer.data();
	}

	void enqueue(
		const feature_type *left,
		const feature_type *right,
		size_t width,
		size_t height,
		unsigned int p1,
		unsigned int p2,
		cudaStream_t stream);

};

}

#endif
