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

#ifndef SGM_DEVICE_BUFFER_HPP
#define SGM_DEVICE_BUFFER_HPP

#include <cstddef>

#include "internal.h"

namespace sgm {

template <typename T>
class DeviceBuffer {

public:
	using value_type = T;

private:
	value_type *m_data;
	size_t m_size;

public:
	DeviceBuffer()
		: m_data(nullptr)
		, m_size(0)
	{ }

	explicit DeviceBuffer(size_t n)
		: m_data(nullptr)
		, m_size(0)
	{
		allocate(n);
	}

	DeviceBuffer(const DeviceBuffer&) = delete;

	DeviceBuffer(DeviceBuffer&& obj)
		: m_data(obj.m_data)
		, m_size(obj.m_size)
	{
		obj.m_data = nullptr;
		obj.m_size = 0;
	}

	~DeviceBuffer(){
		destroy();
	}


	void allocate(size_t n){
		if(m_data && m_size >= n)
			return;

		destroy();
		CudaSafeCall(cudaMalloc(reinterpret_cast<void **>(&m_data), sizeof(value_type) * n));
		m_size = n;
	}

	void destroy(){
		if(m_data)
			CudaSafeCall(cudaFree(m_data));

		m_data = nullptr;
		m_size = 0;
	}

	void fillZero(){
		CudaSafeCall(cudaMemset(m_data, 0, sizeof(value_type) * m_size));
	}

	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	DeviceBuffer& operator=(DeviceBuffer&& obj){
		m_data = obj.m_data;
		m_size = obj.m_size;
		obj.m_data = nullptr;
		obj.m_size = 0;
		return *this;
	}


	size_t size() const {
		return m_size;
	}

	const value_type *data() const {
		return m_data;
	}

	value_type *data(){
		return m_data;
	}

};

}

#endif
