#ifndef SGM_SGM_HPP
#define SGM_SGM_HPP

#include <memory>
#include <cstdint>

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class SemiGlobalMatching {

public:
	using input_type = T;
	using output_type = uint8_t;

private:
	class Impl;
	std::unique_ptr<Impl> m_impl;

public:
	SemiGlobalMatching();
	~SemiGlobalMatching();

	void execute(
		output_type *dest_left,
		output_type *dest_right,
		const input_type *src_left,
		const input_type *src_right,
		size_t width,
		size_t height,
		unsigned int penalty1,
		unsigned int penalty2,
		float uniqueness);

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
		cudaStream_t stream);

};

}

#endif
