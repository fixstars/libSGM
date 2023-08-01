#include <gtest/gtest.h>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"

namespace sgm
{

template <typename T>
static void census_transform_9x7_(const HostImage& src, HostImage& dst)
{
	constexpr int RADIUS_U = 9 / 2;
	constexpr int RADIUS_V = 7 / 2;

	dst.fill_zero();

	for (int v = RADIUS_V; v < src.rows - RADIUS_V; v++) {
		uint64_t* ptrDst = dst.ptr<uint64_t>(v);
		for (int u = RADIUS_U; u < src.cols - RADIUS_U; u++) {
			uint64_t f = 0;
			for (int dv = -RADIUS_V; dv <= RADIUS_V; dv++) {
				for (int du = -RADIUS_U; du <= RADIUS_U; du++) {
					if (du != 0 && dv != 0) {
						f <<= 1;
						f |= (src.ptr<T>(v)[u] > src.ptr<T>(v + dv)[u + du]);
					}
				}
			}
			ptrDst[u] = f;
		}
	}
}

template <typename T>
static void symmetric_census_9x7_(const HostImage& src, HostImage& dst)
{
	constexpr int RADIUS_U = 9 / 2;
	constexpr int RADIUS_V = 7 / 2;

	dst.fill_zero();

	for (int v = RADIUS_V; v < src.rows - RADIUS_V; v++) {
		uint32_t* ptrDst = dst.ptr<uint32_t>(v);
		for (int u = RADIUS_U; u < src.cols - RADIUS_U; u++) {
			uint32_t f = 0;
			for (int dv = -RADIUS_V; dv <= 0; dv++) {
				for (int du = -RADIUS_U; du <= (dv != 0 ? RADIUS_U : -1); du++) {
					f <<= 1;
					f |= (src.ptr<T>(v + dv)[u + du] > src.ptr<T>(v - dv)[u - du]);
				}
			}
			ptrDst[u] = f;
		}
	}
}

void census_transform(const HostImage& src, HostImage& dst, CensusType type)
{
	if (type == CensusType::CENSUS_9x7) {
		dst.create(src.rows, src.cols, SGM_64U);
		if (src.type == SGM_8U)
			census_transform_9x7_<uint8_t>(src, dst);
		if (src.type == SGM_16U)
			census_transform_9x7_<uint16_t>(src, dst);
		if (src.type == SGM_32U)
			census_transform_9x7_<uint32_t>(src, dst);
	}
	if (type == CensusType::SYMMETRIC_CENSUS_9x7) {
		dst.create(src.rows, src.cols, SGM_32U);
		if (src.type == SGM_8U)
			symmetric_census_9x7_<uint8_t>(src, dst);
		if (src.type == SGM_16U)
			symmetric_census_9x7_<uint16_t>(src, dst);
		if (src.type == SGM_32U)
			symmetric_census_9x7_<uint32_t>(src, dst);
	}
}

} // namespace sgm

TEST(CensusTransformTest, RandomU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_64U;
	const CensusType censusType = CensusType::CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(CensusTransformTest, RandomU16)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_64U;
	const CensusType censusType = CensusType::CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(CensusTransformTest, RandomU32)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_32U;
	const ImageType dtype = SGM_64U;
	const CensusType censusType = CensusType::CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(SymmetricCensusTest, RandomU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_32U;
	const CensusType censusType = CensusType::SYMMETRIC_CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(SymmetricCensusTest, Random16U)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_32U;
	const CensusType censusType = CensusType::SYMMETRIC_CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(SymmetricCensusTest, Random32U)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = 640;
	const ImageType stype = SGM_32U;
	const ImageType dtype = SGM_32U;
	const CensusType censusType = CensusType::SYMMETRIC_CENSUS_9x7;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	census_transform(h_src, h_dst, censusType);
	census_transform(d_src, d_dst, censusType);

	EXPECT_TRUE(equals(h_dst, d_dst));
}
