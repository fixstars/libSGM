#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"

struct WinnerTakesAllParam
{
	int disp_size;
	float uniqueness;
	sgm::PathType path_type;
	bool subpixel;
};

static WinnerTakesAllParam params[] = {
	{  64, 0.95f, sgm::PathType::SCAN_4PATH, false },
	{  64, 0.95f, sgm::PathType::SCAN_4PATH, true },
	{  64, 0.95f, sgm::PathType::SCAN_8PATH, false },
	{  64, 0.95f, sgm::PathType::SCAN_8PATH, true },
	{  64, 0.95f, sgm::PathType::SCAN_16PATH, false },
	{  64, 0.95f, sgm::PathType::SCAN_16PATH, true },
	{ 128, 0.95f, sgm::PathType::SCAN_4PATH, false },
	{ 128, 0.95f, sgm::PathType::SCAN_4PATH, true },
	{ 128, 0.95f, sgm::PathType::SCAN_8PATH, false },
	{ 128, 0.95f, sgm::PathType::SCAN_8PATH, true },
	{ 128, 0.95f, sgm::PathType::SCAN_16PATH, false },
	{ 128, 0.95f, sgm::PathType::SCAN_16PATH, true },
	{ 256, 0.95f, sgm::PathType::SCAN_4PATH, false },
	{ 256, 0.95f, sgm::PathType::SCAN_4PATH, true },
	{ 256, 0.95f, sgm::PathType::SCAN_8PATH, false },
	{ 256, 0.95f, sgm::PathType::SCAN_8PATH, true },
	{ 256, 0.95f, sgm::PathType::SCAN_16PATH, false },
	{ 256, 0.95f, sgm::PathType::SCAN_16PATH, true },
};

namespace sgm
{

void winner_takes_all(const HostImage& L, HostImage& D1, HostImage& D2,
	int disp_size, float uniqueness, bool subpixel, PathType path_type)
{
	const int w = D1.cols;
	const int h = D1.rows;
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : path_type == PathType::SCAN_8PATH ? 8 : 16;

	using COST_TYPE = uint8_t;
	using DISP_TYPE = uint16_t;
	using SUM_TYPE = uint32_t;

	constexpr int SUBPIXEL_SCALE = sgm::StereoSGM::SUBPIXEL_SCALE;
	constexpr SUM_TYPE MAX_SUM_COST = std::numeric_limits<SUM_TYPE>::max();

	HostImage costSum(w, disp_size, SGM_32U);

	for (int v = 0; v < h; v++) {

		DISP_TYPE* ptrD1 = D1.ptr<DISP_TYPE>(v);
		DISP_TYPE* ptrD2 = D2.ptr<DISP_TYPE>(v);

		costSum.fill_zero();

		for (int u = 0; u < w; u++) {

			// sum-up costs of each path
			SUM_TYPE* S = costSum.ptr<SUM_TYPE>(u);
			for (int i = 0; i < num_paths; i++) {
				const COST_TYPE* ptrL = L.ptr<COST_TYPE>(i) + (v * w + u) * disp_size;
				for (int k = 0; k < disp_size; k++)
					S[k] += ptrL[k];
			}

			// find disparity with minimum cost
			SUM_TYPE minS = MAX_SUM_COST;
			int disp = 0;
			for (int k = 0; k < disp_size; k++) {
				if (S[k] < minS) {
					minS = S[k];
					disp = k;
				}
			}

			// uniqueness check
			int k;
			for (k = 0; k < disp_size; k++) {
				if (uniqueness * S[k] < S[disp] && std::abs(k - disp) > 1)
					break;
			}
			if (k < disp_size) {
				ptrD1[u] = static_cast<DISP_TYPE>(INVALID_DISP);
				continue;
			}

			// sub-pixel interpolation
			if (subpixel)
			{
				if (disp > 0 && disp < disp_size - 1) {
					const int numer = S[disp - 1] - S[disp + 1];
					const int denom = S[disp - 1] - 2 * S[disp] + S[disp + 1];
					disp = disp * SUBPIXEL_SCALE + (SUBPIXEL_SCALE * numer + denom) / (2 * denom);
				}
				else {
					disp *= SUBPIXEL_SCALE;
				}
			}

			ptrD1[u] = static_cast<DISP_TYPE>(disp);
		}

		// calculate right disparity
		for (int u = 0; u < w; u++) {
			SUM_TYPE minS = MAX_SUM_COST;
			int disp = 0;
			for (int k = 0; k < disp_size && u + k < w; k++) {
				const SUM_TYPE S = costSum.ptr<SUM_TYPE>(u + k)[k];
				if (S < minS) {
					minS = S;
					disp = k;
				}
			}
			ptrD2[u] = static_cast<DISP_TYPE>(disp);
		}
	}
}

} // namespace sgm

class WinnerTakesAllTestP : public ::testing::TestWithParam<WinnerTakesAllParam> {};
INSTANTIATE_TEST_CASE_P(TestDataIntRange, WinnerTakesAllTestP, ::testing::ValuesIn(params));

TEST_P(WinnerTakesAllTestP, RangeTest)
{
	using namespace sgm;
	using namespace details;

	//GTEST_SKIP();

	const auto param = GetParam();

	const int w = 311;
	const int h = 239;
	const int pitch = 320;
	const int disp_size = param.disp_size;
	const int num_paths = param.path_type == PathType::SCAN_4PATH ? 4 : param.path_type == PathType::SCAN_8PATH ? 8 : 16;
	const auto cost_type = SGM_8U;
	const auto disp_type = SGM_16U;

	HostImage h_cost(num_paths, w * h * disp_size, cost_type);
	HostImage h_dispL(h, w, disp_type, pitch), h_dispR(h, w, disp_type, pitch);

	DeviceImage d_cost(num_paths, w * h * disp_size, cost_type);
	DeviceImage d_dispL(h, w, disp_type, pitch), d_dispR(h, w, disp_type, pitch);

	random_fill(h_cost);
	d_cost.upload(h_cost.data);

	winner_takes_all(h_cost, h_dispL, h_dispR, disp_size, param.uniqueness, param.subpixel, param.path_type);
	winner_takes_all(d_cost, d_dispL, d_dispR, disp_size, param.uniqueness, param.subpixel, param.path_type);

	EXPECT_TRUE(equals(h_dispL, d_dispL));
	EXPECT_TRUE(equals(h_dispR, d_dispR));
}
