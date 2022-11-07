#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"

namespace sgm
{

void correct_disparity_range(HostImage& disp, bool subpixel, int min_disp)
{
	const int h = disp.rows;
	const int w = disp.cols;

	const int scale = subpixel ? StereoSGM::SUBPIXEL_SCALE : 1;
	const int     min_disp_scaled =  min_disp      * scale;
	const int invalid_disp_scaled = (min_disp - 1) * scale;

	for (int y = 0; y < h; y++)
	{
		uint16_t* ptrDisp = disp.ptr<uint16_t>(y);
		for (int x = 0; x < w; x++)
		{
			uint16_t d = ptrDisp[x];
			if (d == sgm::INVALID_DISP) {
				d = invalid_disp_scaled;
			}
			else {
				d += min_disp_scaled;
			}
			ptrDisp[x] = d;
		}
	}
}

} // namespace sgm

using Parameters = std::tuple<int, int, int>;

class CorrectDisparityRangeTest : public ::testing::TestWithParam<Parameters> {};
INSTANTIATE_TEST_CASE_P(TestWithParams, CorrectDisparityRangeTest,
	::testing::Combine(::testing::Values(64, 128, 256), ::testing::Values(0, 1), ::testing::Values(0, +16, -16)));

TEST_P(CorrectDisparityRangeTest, Random16U)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType dtype = SGM_16U;

	const auto param = GetParam();
	const int disp_size = std::get<0>(param);
	const bool subpixel = std::get<1>(param) > 0;
	const bool min_disp = std::get<2>(param);

	HostImage h_disp(h, w, dtype, pitch);
	DeviceImage d_disp(h, w, dtype, pitch);

	random_fill(h_disp, 0, disp_size);
	d_disp.upload(h_disp.data);

	correct_disparity_range(h_disp, subpixel, min_disp);
	correct_disparity_range(d_disp, subpixel, min_disp);

	EXPECT_TRUE(equals(h_disp, d_disp));
}
