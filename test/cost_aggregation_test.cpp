#include <gtest/gtest.h>

#include <algorithm>

#include "host_image.h"
#include "device_image.h"
#include "test_utility.h"
#include "internal.h"
#include "constants.h"

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

struct CostAggregationParam
{
	sgm::ImageType census_type;
	int disp_size;
	int P1, P2;
	int min_disp;
};

static CostAggregationParam params[] = {
	{ sgm::SGM_32U,  64, 10, 120,  +0 },
	{ sgm::SGM_32U,  64, 10, 120, +16 },
	{ sgm::SGM_32U,  64, 10, 120, -16 },
	{ sgm::SGM_32U, 128, 10, 120,  +0 },
	{ sgm::SGM_32U, 128, 10, 120, +16 },
	{ sgm::SGM_32U, 128, 10, 120, -16 },
	{ sgm::SGM_32U, 256, 10, 120,  +0 },
	{ sgm::SGM_32U, 256, 10, 120, +16 },
	{ sgm::SGM_32U, 256, 10, 120, -16 },
	{ sgm::SGM_64U,  64, 10, 120,  +0 },
	{ sgm::SGM_64U,  64, 10, 120, +16 },
	{ sgm::SGM_64U,  64, 10, 120, -16 },
	{ sgm::SGM_64U, 128, 10, 120,  +0 },
	{ sgm::SGM_64U, 128, 10, 120, +16 },
	{ sgm::SGM_64U, 128, 10, 120, -16 },
	{ sgm::SGM_64U, 256, 10, 120,  +0 },
	{ sgm::SGM_64U, 256, 10, 120, +16 },
	{ sgm::SGM_64U, 256, 10, 120, -16 },
};

namespace sgm
{

using COST_TYPE = uint8_t;

static inline int HammingDistance(uint64_t c1, uint64_t c2) { return static_cast<int>(popcnt64(c1 ^ c2)); }
static inline int HammingDistance(uint32_t c1, uint32_t c2) { return static_cast<int>(popcnt32(c1 ^ c2)); }

static inline int min4(int x, int y, int z, int w)
{
	return std::min(std::min(x, y), std::min(z, w));
};

template <typename CENSUS_TYPE>
static void cost_aggregation_(const HostImage& srcL, const HostImage& srcR, HostImage& dst,
	int disp_size, int P1, int P2, int min_disp, int ru, int rv)
{
	const int h = srcL.rows;
	const int w = srcL.cols;
	const int n = disp_size;

	const bool forward = rv > 0 || (rv == 0 && ru > 0);
	int u0 = 0, u1 = w, du = 1, v0 = 0, v1 = h, dv = 1;
	if (!forward) {
		u0 = w - 1; u1 = -1; du = -1;
		v0 = h - 1; v1 = -1; dv = -1;
	}

	std::vector<COST_TYPE> zero(disp_size, 0);

	for (int vc = v0; vc != v1; vc += dv) {

		const CENSUS_TYPE* censusL = srcL.ptr<CENSUS_TYPE>(vc);
		const CENSUS_TYPE* censusR = srcR.ptr<CENSUS_TYPE>(vc);
		for (int uc = u0; uc != u1; uc += du) {

			const int vp = vc - rv;
			const int up = uc - ru;
			const bool inside = vp >= 0 && vp < h&& up >= 0 && up < w;

			const CENSUS_TYPE cL = censusL[uc];
			COST_TYPE* Lc = dst.ptr<COST_TYPE>(vc * w + uc);
			COST_TYPE* Lp = inside ? dst.ptr<COST_TYPE>(vp * w + up) : zero.data();

			COST_TYPE minLp = std::numeric_limits<COST_TYPE>::max();
			for (int d = 0; d < n; d++)
				minLp = std::min(minLp, Lp[d]);

			const COST_TYPE _P1 = P1 - minLp;
			for (int d = 0; d < n; d++) {
				const int uR = uc - d - min_disp;
				const CENSUS_TYPE cR = uR >= 0 && uR < w ? censusR[uR] : 0;
				const COST_TYPE MC = HammingDistance(cL, cR);
				const COST_TYPE Lp0 = Lp[d] - minLp;
				const COST_TYPE Lp1 = d > 0 ?     Lp[d - 1] + _P1 : 0xFF;
				const COST_TYPE Lp2 = d < n - 1 ? Lp[d + 1] + _P1 : 0xFF;
				const COST_TYPE Lp3 = P2;
				Lc[d] = static_cast<COST_TYPE>(MC + min4(Lp0, Lp1, Lp2, Lp3));
			}
		}
	}
}

static void cost_aggregation(const HostImage& srcL, const HostImage& srcR, HostImage& dst,
	int disp_size, int P1, int P2, int min_disp, int ru, int rv)
{
	if (srcL.type == SGM_32U)
		cost_aggregation_<uint32_t>(srcL, srcR, dst, disp_size, P1, P2, min_disp, ru, rv);
	if (srcL.type == SGM_64U)
		cost_aggregation_<uint64_t>(srcL, srcR, dst, disp_size, P1, P2, min_disp, ru, rv);
}

void cost_aggregation(const HostImage& srcL, const HostImage& srcR, HostImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp)
{
	const int MAX_DIRECTIONS = 8;
	const int ru[MAX_DIRECTIONS] = { +0, +0, +1, -1, +1, -1, -1, +1 };
	const int rv[MAX_DIRECTIONS] = { +1, -1, +0, +0, +1, +1, -1, -1 };

	const int w = srcL.cols;
	const int h = srcL.rows;
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	dst.create(num_paths, h * w * disp_size, SGM_8U);

	for (int i = 0; i < num_paths; i++)
	{
		HostImage cost(dst.ptr<COST_TYPE>(i), h * w, disp_size, SGM_8U);
		cost_aggregation(srcL, srcR, cost, disp_size, P1, P2, min_disp, ru[i], rv[i]);
	}
}

} // namespace sgm

class CostAggregationTest : public ::testing::TestWithParam<CostAggregationParam> {};
INSTANTIATE_TEST_CASE_P(TestWithParams, CostAggregationTest, ::testing::ValuesIn(params));

TEST_P(CostAggregationTest, AllPathsTest)
{
	using namespace sgm;
	using namespace details;

	//GTEST_SKIP();

	const auto param = GetParam();

	const int w = 320;
	const int h = 240;
	const int disp_size = param.disp_size;
	const auto path_type = PathType::SCAN_8PATH;
	const int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;
	const int P1 = param.P1;
	const int P2 = param.P2;
	const int min_disp = param.min_disp;

	const ImageType census_type = param.census_type;
	const ImageType cost_type = SGM_8U;

	HostImage h_censusL(h, w, census_type), h_censusR(h, w, census_type);
	HostImage h_costs;

	DeviceImage d_censusL(h, w, census_type), d_censusR(h, w, census_type);
	DeviceImage d_costs;

	random_fill(h_censusL);
	random_fill(h_censusR);
	d_censusL.upload(h_censusL.data);
	d_censusR.upload(h_censusR.data);

	cost_aggregation(h_censusL, h_censusR, h_costs, disp_size, P1, P2, path_type, min_disp);
	cost_aggregation(d_censusL, d_censusR, d_costs, disp_size, P1, P2, path_type, min_disp);

	for (int i = 0; i < num_paths; i++) {
		HostImage h_cost(h_costs.ptr<COST_TYPE>(i), h * w, disp_size, cost_type);
		DeviceImage d_cost(d_costs.ptr<COST_TYPE>(i), h * w, disp_size, cost_type);
		EXPECT_TRUE(equals(h_cost, d_cost));
	}
}
