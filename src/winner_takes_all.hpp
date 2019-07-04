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

#ifndef SGM_WINNER_TAKES_ALL_HPP
#define SGM_WINNER_TAKES_ALL_HPP

#include <libsgm.h>
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
		int width,
		int height,
		int pitch,
		float uniqueness,
		bool subpixel,
		PathType path_type,
		cudaStream_t stream);

	void enqueue(
		output_type *left,
		output_type *right,
		const cost_type *src,
		int width,
		int height,
		int pitch,
		float uniqueness,
		bool subpixel,
		PathType path_type,
		cudaStream_t stream);

};

}

#endif
