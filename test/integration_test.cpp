#include <gtest/gtest.h>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"
#include "reference.h"

TEST(IntegrationTest, RandomU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 311;
	const int h = 239;
	const int pitch = 320;
	const int disp_size = 128;
	const int P1 = 10;
	const int P2 = 120;
	const float uniqueness = 0.95f;
	const auto path_type = PathType::SCAN_4PATH;
	const int min_disp = 0;
	const bool subpixel = true;
	const int LR_max_diff = 5;
	const auto censusType = CensusType::SYMMETRIC_CENSUS_9x7;

	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_16U;
	const ImageType ctype = censusType == CensusType::CENSUS_9x7 ? SGM_64U : SGM_32U;

	HostImage h_srcL(h, w, stype, pitch), h_srcR(h, w, stype, pitch);
	DeviceImage d_srcL(h, w, stype, pitch), d_srcR(h, w, stype, pitch);

	HostImage h_censusL(h, w, ctype), h_censusR(h, w, ctype), h_costs;
	DeviceImage d_censusL(h, w, ctype), d_censusR(h, w, ctype), d_costs;

	HostImage h_tmpL(h, w, dtype), h_tmpR(h, w, dtype), h_dispL(h, w, dtype), h_dispR(h, w, dtype);
	DeviceImage d_tmpL(h, w, dtype), d_tmpR(h, w, dtype), d_dispL(h, w, dtype), d_dispR(h, w, dtype);

	random_fill(h_srcL);
	random_fill(h_srcR);
	d_srcL.upload(h_srcL.data);
	d_srcR.upload(h_srcR.data);
	d_censusL.fill_zero();
	d_censusR.fill_zero();
	d_dispL.fill_zero();
	d_dispR.fill_zero();

	// census transform
	census_transform(h_srcL, h_censusL, censusType);
	census_transform(h_srcR, h_censusR, censusType);

	census_transform(d_srcL, d_censusL, censusType);
	census_transform(d_srcR, d_censusR, censusType);
	
	EXPECT_TRUE(equals(h_censusL, d_censusL));
	EXPECT_TRUE(equals(h_censusR, d_censusR));

	// cost aggregation
	cost_aggregation(h_censusL, h_censusR, h_costs, disp_size, P1, P2, path_type, min_disp);
	cost_aggregation(d_censusL, d_censusR, d_costs, disp_size, P1, P2, path_type, min_disp);
	EXPECT_TRUE(equals(h_costs, d_costs));

	// winner takes all
	winner_takes_all(h_costs, h_tmpL, h_tmpR, disp_size, uniqueness, subpixel, path_type);
	winner_takes_all(d_costs, d_tmpL, d_tmpR, disp_size, uniqueness, subpixel, path_type);
	EXPECT_TRUE(equals(h_tmpL, d_tmpL));
	EXPECT_TRUE(equals(h_tmpR, d_tmpR));

	// post filtering
	median_filter(h_tmpL, h_dispL);
	median_filter(d_tmpL, d_dispL);
	EXPECT_TRUE(equals(h_dispL, d_dispL));

	median_filter(h_tmpR, h_dispR);
	median_filter(d_tmpR, d_dispR);
	EXPECT_TRUE(equals(h_dispR, d_dispR));

	// consistency check
	check_consistency(h_dispL, h_dispR, h_srcL, subpixel, LR_max_diff);
	check_consistency(d_dispL, d_dispR, d_srcL, subpixel, LR_max_diff);
	EXPECT_TRUE(equals(h_dispL, d_dispL));
}
