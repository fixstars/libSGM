#ifndef __HOST_IMAGE_H__
#define __HOST_IMAGE_H__

#include <cstdlib>

#include "device_image.h"

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

class HostImage
{
public:

	HostImage() : data(nullptr), rows(0), cols(0), step(0), type(SGM_8U), allocated(false)
	{
	}

	HostImage(int _rows, int _cols, ImageType _type, int _step = -1) : HostImage()
	{
		create(_rows, _cols, _type, _step);
	}

	HostImage(void* _data, int _rows, int _cols, ImageType _type, int _step = -1) : HostImage()
	{
		create(_data, _rows, _cols, _type, _step);
	}

	~HostImage()
	{
		release();
	}

	void create(int _rows, int _cols, ImageType _type, int _step = -1)
	{
		release();

		if (_step < 0)
			_step = _cols;

		data = malloc(elemSize(_type) * _rows * _step);
		rows = _rows;
		cols = _cols;
		step = _step;
		type = _type;
		allocated = true;
	}

	void create(void* _data, int _rows, int _cols, ImageType _type, int _step = -1)
	{
		release();

		if (_step < 0)
			_step = _cols;

		data = _data;
		rows = _rows;
		cols = _cols;
		step = _step;
		type = _type;
		allocated = false;
	}

	void release()
	{
		if (allocated && data)
			free(data);
		data = nullptr;
		rows = cols = step = 0;
		allocated = false;
	}

	void fill_zero()
	{
		memset(data, 0, elemSize(type) * rows * step);
	}

	void copy_to(HostImage& rhs) const
	{
		rhs.create(rows, cols, type, step);
		memcpy(rhs.data, data, elemSize(type) * rows * step);
	}

	template <typename T> T* ptr(int y = 0) { return (T*)data + y * step; }
	template <typename T> const T* ptr(int y = 0) const { return (T*)data + y * step; }

	void* data;
	int rows, cols, step;
	ImageType type;
	bool allocated;
};

} // namespace sgm

#endif // !__HOST_IMAGE_H__
