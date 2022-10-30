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

#include "device_image.h"

#include <cuda_runtime.h>

#include "host_utility.h"

namespace sgm
{

static size_t elemSize(ImageType type)
{
	if (type == SGM_8U)
		return 1;
	if (type == SGM_16U)
		return 2;
	if (type == SGM_32U)
		return 4;
	if (type == SGM_64U)
		return 8;
	return 0;
}

DeviceImage::DeviceImage() : data(nullptr), rows(0), cols(0), step(0), type(SGM_8U)
{
}

DeviceImage::DeviceImage(int rows, int cols, ImageType type, int step)
{
	create(rows, cols, type, step);
}

DeviceImage::DeviceImage(void* data, int rows, int cols, ImageType type, int step)
{
	create(data, rows, cols, type, step);
}

void DeviceImage::create(int _rows, int _cols, ImageType _type, int _step)
{
	if (_step < 0)
		_step = _cols;

	data = allocator_.allocate(elemSize(_type) * _rows * _step);
	rows = _rows;
	cols = _cols;
	step = _step;
	type = _type;
}

void DeviceImage::create(void* _data, int _rows, int _cols, ImageType _type, int _step)
{
	if (_step < 0)
		_step = _cols;

	allocator_.assign(_data, elemSize(_type) * _rows * _step);
	data = _data;
	rows = _rows;
	cols = _cols;
	step = _step;
	type = _type;
}

void DeviceImage::upload(const void* _data)
{
	CUDA_CHECK(cudaMemcpy(data, _data, elemSize(type) * rows * step, cudaMemcpyHostToDevice));
}

void DeviceImage::download(void* _data) const
{
	CUDA_CHECK(cudaMemcpy(_data, data, elemSize(type) * rows * step, cudaMemcpyDeviceToHost));
}

void DeviceImage::fill_zero()
{
	CUDA_CHECK(cudaMemset(data, 0, elemSize(type) * rows * step));
}

} // namespace sgm
