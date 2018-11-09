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

#ifndef SGM_CENSUS_TRANSFORM_HPP
#define SGM_CENSUS_TRANSFORM_HPP

#include "device_buffer.hpp"
#include "types.hpp"

namespace sgm {

template <typename T>
class CensusTransform {

public:
	using input_type = T;

private:
	DeviceBuffer<feature_type> m_feature_buffer;

public:
	CensusTransform();

	const feature_type *get_output() const {
		return m_feature_buffer.data();
	}
	
	void enqueue(
		const input_type *src,
		int width,
		int height,
		int pitch,
		cudaStream_t stream);

};

}

#endif
