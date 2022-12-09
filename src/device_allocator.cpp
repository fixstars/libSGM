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

#include "device_allocator.h"

#include <cuda_runtime.h>

#include "host_utility.h"

namespace sgm
{

DeviceAllocator::DeviceAllocator() : data_(nullptr), refCount_(nullptr), capacity_(0)
{
}

DeviceAllocator::DeviceAllocator(const DeviceAllocator& other)
{
	copy_construct_from(other);
}

DeviceAllocator::DeviceAllocator(DeviceAllocator&& right)
{
	move_construct_from(std::move(right));
}

DeviceAllocator::~DeviceAllocator()
{
	release();
}

void* DeviceAllocator::allocate(size_t size)
{
	if (size > capacity_)
	{
		release();
		CUDA_CHECK(cudaMalloc(&data_, size));
		refCount_ = new int(1);
		capacity_ = size;
	}
	return data_;
}

void DeviceAllocator::assign(void* data, size_t size)
{
	release();
	data_ = data;
	capacity_ = size;
}

void DeviceAllocator::release()
{
	if (refCount_ && --(*refCount_) == 0)
	{
		CUDA_CHECK(cudaFree(data_));
		delete refCount_;
	}

	data_ = refCount_ = nullptr;
	capacity_ = 0;
}

DeviceAllocator& DeviceAllocator::operator=(const DeviceAllocator& other)
{
	release();
	copy_construct_from(other);
	return *this;
}

DeviceAllocator& DeviceAllocator::operator=(DeviceAllocator&& right)
{
	release();
	move_construct_from(std::move(right));
	return *this;
}

void DeviceAllocator::copy_construct_from(const DeviceAllocator& other)
{
	data_ = other.data_;
	refCount_ = other.refCount_;
	capacity_ = other.capacity_;

	if (refCount_)
		(*refCount_)++;
}

void DeviceAllocator::move_construct_from(DeviceAllocator&& right)
{
	data_ = right.data_;
	refCount_ = right.refCount_;
	capacity_ = right.capacity_;

	right.data_ = right.refCount_ = nullptr;
	right.capacity_ = 0;
}

} // namespace sgm
