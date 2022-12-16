#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"

namespace sgm
{

void cast_16bit_to_8bit(const HostImage& src, HostImage& dst)
{
	const int h = src.rows;
	const int w = dst.cols;

	dst.create(h, w, SGM_8U);

	for (int y = 0; y < h; y++)
	{
		const uint16_t* ptrSrc = src.ptr<uint16_t>(y);
		uint8_t* ptrDst = dst.ptr<uint8_t>(y);
		for (int x = 0; x < w; x++)
			ptrDst[x] = static_cast<uint8_t>(ptrSrc[x]);
	}
}

void cast_8bit_to_16bit(const HostImage& src, HostImage& dst)
{
	const int h = src.rows;
	const int w = dst.cols;

	dst.create(h, w, SGM_16U);

	for (int y = 0; y < h; y++)
	{
		const uint8_t* ptrSrc = src.ptr<uint8_t>(y);
		uint16_t* ptrDst = dst.ptr<uint16_t>(y);
		for (int x = 0; x < w; x++)
			ptrDst[x] = static_cast<uint16_t>(ptrSrc[x]);
	}
}

} // namespace sgm

TEST(CastTest, RandomU16ToU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_8U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);

	cast_16bit_to_8bit(h_src, h_dst);
	cast_16bit_to_8bit(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(CastTest, RandomU8ToU16)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_16U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);

	cast_8bit_to_16bit(h_src, h_dst);
	cast_8bit_to_16bit(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}
