#ifndef __TEST_UTILITY_H__
#define __TEST_UTILITY_H__

#include <random>
#include <limits>

#include "host_image.h"

static std::default_random_engine g_engine;

template <typename T> constexpr T min_of() { return std::numeric_limits<T>::min(); }
template <> constexpr uint8_t min_of() { return 0; }
template <typename T> constexpr T max_of() { return std::numeric_limits<T>::max(); }
template <> constexpr uint8_t max_of() { return 255; }

template <typename T>
static void random_fill_(T* dst, int n, T minv = min_of<T>(), T maxv = max_of<T>())
{
	std::uniform_int_distribution<T> dist(minv, maxv);
	for (int i = 0; i < n; ++i)
		dst[i] = dist(g_engine);
}

template <>
static void random_fill_(uint8_t* dst, int n, uint8_t minv, uint8_t maxv)
{
	std::uniform_int_distribution<uint32_t> dist(minv, maxv);
	for (int i = 0; i < n; ++i)
		dst[i] = static_cast<uint8_t>(dist(g_engine));
}

template <typename T>
static void random_fill_(sgm::HostImage& image, T minv = min_of<T>(), T maxv = max_of<T>())
{
	random_fill_(image.ptr<T>(), image.rows * image.step, minv, maxv);
}

static void random_fill(sgm::HostImage& image)
{
	if (image.type == sgm::SGM_8U)
		random_fill_<uint8_t>(image);
	if (image.type == sgm::SGM_16U)
		random_fill_<uint16_t>(image);
	if (image.type == sgm::SGM_32U)
		random_fill_<uint32_t>(image);
	if (image.type == sgm::SGM_64U)
		random_fill_<uint64_t>(image);
}

static void random_fill(sgm::HostImage& image, int minv, int maxv)
{
	if (image.type == sgm::SGM_8U)
		random_fill_<uint8_t>(image, minv, maxv);
	if (image.type == sgm::SGM_16U)
		random_fill_<uint16_t>(image, minv, maxv);
	if (image.type == sgm::SGM_32U)
		random_fill_<uint32_t>(image, minv, maxv);
	if (image.type == sgm::SGM_64U)
		random_fill_<uint64_t>(image, minv, maxv);
}

template <typename T>
static int count_nonzero_(const sgm::HostImage& a, const sgm::HostImage& b)
{
	if (a.cols != b.cols || a.rows != b.rows)
		return -1;

	int count = 0;
	for (int y = 0; y < a.rows; y++)
	{
		const T* pa = a.ptr<T>(y);
		const T* pb = b.ptr<T>(y);
		for (int x = 0; x < a.cols; x++)
			if (pa[x] != pb[x])
				count++;
	}
	return count;
}

static int count_nonzero(const sgm::HostImage& a, const sgm::HostImage& b)
{
	if (a.type != b.type)
		return -1;

	if (a.type == sgm::SGM_8U)
		return count_nonzero_<uint8_t>(a, b);
	if (a.type == sgm::SGM_16U)
		return count_nonzero_<uint16_t>(a, b);
	if (a.type == sgm::SGM_32U)
		return count_nonzero_<uint32_t>(a, b);
	if (a.type == sgm::SGM_64U)
		return count_nonzero_<uint64_t>(a, b);

	return -1;
}

template <typename T>
static bool equals_(const sgm::HostImage& a, const sgm::HostImage& b)
{
	if (a.cols != b.cols || a.rows != b.rows)
		return false;

	for (int y = 0; y < a.rows; y++)
	{
		const T* pa = a.ptr<T>(y);
		const T* pb = b.ptr<T>(y);
		for (int x = 0; x < a.cols; x++)
			if (pa[x] != pb[x])
				return false;
	}
	return true;
}

static bool equals(const sgm::HostImage& a, const sgm::HostImage& b)
{
	if (a.type != b.type)
		return false;

	if (a.type == sgm::SGM_8U)
		return equals_<uint8_t>(a, b);
	if (a.type == sgm::SGM_16U)
		return equals_<uint16_t>(a, b);
	if (a.type == sgm::SGM_32U)
		return equals_<uint32_t>(a, b);
	if (a.type == sgm::SGM_64U)
		return equals_<uint64_t>(a, b);

	return false;
}

static bool equals(const sgm::HostImage& h_a, const sgm::DeviceImage& d_b)
{
	if (h_a.type != d_b.type || h_a.rows != d_b.rows || h_a.cols != d_b.cols)
		return false;

	sgm::HostImage h_b(d_b.rows, d_b.cols, d_b.type, d_b.step);
	d_b.download(h_b.data);
	return equals(h_a, h_b);
}

#endif // !__TEST_UTILITY_H__
