#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"

namespace sgm
{

template <typename T> inline void swap_op(T& x, T& y) { if (x > y) std::swap(x, y); }
template <typename T> inline void min_op(T& x, T& y) { x = std::min(x, y); }
template <typename T> inline void max_op(T& x, T& y) { y = std::max(x, y); }

template <typename T>
static inline T median_selection_network_9(T* buf)
{
#define SWAP_OP(i, j) swap_op(buf[i], buf[j])
#define MIN_OP(i, j) min_op(buf[i], buf[j])
#define MAX_OP(i, j) max_op(buf[i], buf[j])

	SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
	SWAP_OP(1, 2); SWAP_OP(4, 5); SWAP_OP(7, 8);
	SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
	MAX_OP(0, 3); MAX_OP(3, 6);
	SWAP_OP(1, 4); MIN_OP(4, 7); MAX_OP(1, 4);
	MIN_OP(5, 8); MIN_OP(2, 5);
	SWAP_OP(2, 4); MIN_OP(4, 6); MAX_OP(2, 4);

	return buf[9 / 2];

#undef SWAP_OP
#undef MIN_OP
#undef MAX_OP
}

template <typename T>
static void median_filter_(const HostImage& src, HostImage& dst)
{
	dst.fill_zero();

	T buf[9];

	for (int y = 1; y < src.rows - 1; y++)
	{
		for (int x = 1; x < src.cols - 1; x++)
		{
			int n = 0;
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++)
					buf[n++] = src.ptr<T>(y + dy)[x + dx];

			dst.ptr<T>(y)[x] = median_selection_network_9(buf);
		}
	}
}

void median_filter(const HostImage& src, HostImage& dst)
{
	if (src.type == SGM_8U)
		median_filter_<uint8_t>(src, dst);
	if (src.type == SGM_16U)
		median_filter_<uint16_t>(src, dst);
}

} // namespace sgm

TEST(MedianFilterTest, RandomU8)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = w;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_8U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	median_filter(h_src, h_dst);
	median_filter(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(MedianFilterTest, RandomU8v4)
{
	using namespace sgm;
	using namespace details;

	const int w = 642;
	const int h = 479;
	const int pitch = 644;
	const ImageType stype = SGM_8U;
	const ImageType dtype = SGM_8U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	median_filter(h_src, h_dst);
	median_filter(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(MedianFilterTest, RandomU16)
{
	using namespace sgm;
	using namespace details;

	const int w = 631;
	const int h = 479;
	const int pitch = w;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_16U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	median_filter(h_src, h_dst);
	median_filter(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}

TEST(MedianFilterTest, RandomU16v2)
{
	using namespace sgm;
	using namespace details;

	const int w = 641;
	const int h = 479;
	const int pitch = 644;
	const ImageType stype = SGM_16U;
	const ImageType dtype = SGM_16U;

	HostImage h_src(h, w, stype, pitch), h_dst(h, w, dtype, pitch);
	DeviceImage d_src(h, w, stype, pitch), d_dst(h, w, dtype, pitch);

	random_fill(h_src);
	d_src.upload(h_src.data);
	d_dst.fill_zero();

	median_filter(h_src, h_dst);
	median_filter(d_src, d_dst);

	EXPECT_TRUE(equals(h_dst, d_dst));
}
