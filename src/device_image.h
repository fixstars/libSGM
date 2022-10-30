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

#ifndef __DEVICE_IMAGE_H__
#define __DEVICE_IMAGE_H__

#include "device_allocator.h"

namespace sgm
{

enum ImageType
{
	SGM_8U,
	SGM_16U,
	SGM_32U,
	SGM_64U,
};

class DeviceImage
{
public:

	DeviceImage();
	DeviceImage(int rows, int cols, ImageType type, int step = -1);
	DeviceImage(void* data, int rows, int cols, ImageType type, int step = -1);

	void create(int rows, int cols, ImageType type, int step = -1);
	void create(void* data, int rows, int cols, ImageType type, int step = -1);

	void upload(const void* data);
	void download(void* data) const;
	void fill_zero();

	template <typename T> T* ptr(int y = 0) { return (T*)data + y * step; }
	template <typename T> const T* ptr(int y = 0) const { return (T*)data + y * step; }

	void* data;
	int rows, cols, step;
	ImageType type;

private:

	DeviceAllocator allocator_;
};

} // namespace sgm

#endif // !__DEVICE_IMAGE_H__
