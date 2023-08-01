#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"

namespace sgm
{

template <typename SRC_T>
static void check_consistency_(HostImage& dispL, const HostImage& dispR, const HostImage& srcL,
	bool subpixel, int LR_max_diff)
{
	using DST_T = uint16_t;

	const int h = srcL.rows;
	const int w = srcL.cols;

	for (int y = 0; y < h; y++)
	{
		const SRC_T* ptrMask = srcL.ptr<SRC_T>(y);
		DST_T* ptrDispL = dispL.ptr<DST_T>(y);
		const DST_T* ptrDispR = dispR.ptr<DST_T>(y);
		for (int x = 0; x < w; x++)
		{
			const SRC_T mask = ptrMask[x];
			const DST_T disp = ptrDispL[x];
			int d = disp;
			if (subpixel) {
				d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
			}
			const int k = x - d;
			if (mask == 0 || disp == sgm::INVALID_DISP ||
				(k >= 0 && k < w && LR_max_diff >= 0 && abs(ptrDispR[k] - d) > LR_max_diff)) {
				ptrDispL[x] = static_cast<DST_T>(sgm::INVALID_DISP);
			}
		}
	}
}

void check_consistency(HostImage& dispL, const HostImage& dispR, const HostImage& srcL,
	bool subpixel, int LR_max_diff)
{
	if (srcL.type == SGM_8U)
		check_consistency_<uint8_t>(dispL, dispR, srcL, subpixel, LR_max_diff);
	if (srcL.type == SGM_16U)
		check_consistency_<uint16_t>(dispL, dispR, srcL, subpixel, LR_max_diff);
	if (srcL.type == SGM_32U)
		check_consistency_<uint32_t>(dispL, dispR, srcL, subpixel, LR_max_diff);
}

} // namespace sgm

TEST(CheckConsistencyTest, RandomU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = false;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}

TEST(CheckConsistencyTest, RandomU16)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = false;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}

TEST(CheckConsistencyTest, RandomU32)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_32U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = false;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}

TEST(CheckConsistencyTest, RandomU8_Subpixel)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = true;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}

TEST(CheckConsistencyTest, RandomU16_Subpixel)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = true;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}

TEST(CheckConsistencyTest, RandomU32_Subpixel)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_32U;
	const ImageType dtype = SGM_16U;
	const int LR_max_diff = 5;
	const bool subpixel = true;

	HostImage h_srcL(h, w, stype, pitch), h_dispL(h, w, dtype, pitch), h_dispR(h, w, dtype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_dispL(h, w, dtype, pitch), d_dispR(h, w, dtype, pitch);

	random_fill(h_srcL);
	random_fill(h_dispL);
	random_fill(h_dispR);

	d_srcL.upload(h_srcL.data);
	d_dispL.upload(h_dispL.data);
	d_dispR.upload(h_dispR.data);

	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
}
