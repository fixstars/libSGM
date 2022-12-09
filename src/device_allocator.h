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

#ifndef __DEVICE_ALLOCATOR_H__
#define __DEVICE_ALLOCATOR_H__

#include <cstddef>

namespace sgm
{

class DeviceAllocator
{
public:

	DeviceAllocator();
	DeviceAllocator(const DeviceAllocator& other);
	DeviceAllocator(DeviceAllocator&& right);
	~DeviceAllocator();
	void* allocate(size_t size);
	void assign(void* data, size_t size);
	void release();

	DeviceAllocator& operator=(const DeviceAllocator& other);
	DeviceAllocator& operator=(DeviceAllocator&& right);

private:

	void copy_construct_from(const DeviceAllocator& other);
	void move_construct_from(DeviceAllocator&& right);

	void* data_;
	int* refCount_;
	size_t capacity_;
};

} // namespace sgm

#endif // !__DEVICE_ALLOCATOR_H__
