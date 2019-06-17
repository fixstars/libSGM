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

#ifndef SGM_SGM_HPP
#define SGM_SGM_HPP

#include <memory>
#include <cstdint>
#include <libsgm.h>
#include "types.hpp"

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class SemiGlobalMatching {

public:
	using input_type = T;
	using output_type = sgm::output_type;

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
		int width,
		int height,
		int src_pitch,
		int dst_pitch,
		const StereoSGM::Parameters& param);

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
		cudaStream_t stream);

};

}

#endif
