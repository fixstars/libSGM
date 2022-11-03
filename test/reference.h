#ifndef __REFERENCE_H__
#define __REFERENCE_H__

#include "libsgm.h"
#include "host_image.h"

namespace sgm
{

void census_transform(const HostImage& src, HostImage& dst, CensusType type);
void cost_aggregation(const HostImage& srcL, const HostImage& srcR, HostImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp);
void winner_takes_all(const HostImage& src, HostImage& dstL, HostImage& dstR,
	int disp_size, float uniqueness, bool subpixel, PathType path_type);
void median_filter(const HostImage& src, HostImage& dst);
void check_consistency(HostImage& dispL, const HostImage& dispR, const HostImage& srcL, bool subpixel, int LR_max_diff);

} // namespace sgm

#endif // !__REFERENCE_H__
