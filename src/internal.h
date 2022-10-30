/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "libsgm.h"
#include "device_image.h"

namespace sgm
{
namespace details
{

void census_transform(const DeviceImage& src, DeviceImage& dst);

void cost_aggregation(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp);

void winner_takes_all(const DeviceImage& src, DeviceImage& dstL, DeviceImage& dstR,
	int disp_size, float uniqueness, bool subpixel, PathType path_type);

void median_filter(const DeviceImage& src, DeviceImage& dst);

void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff);

void correct_disparity_range(DeviceImage& disp, bool subpixel, int min_disp);

void cast_16bit_to_8bit(const DeviceImage& src, DeviceImage& dst);
void cast_8bit_to_16bit(const DeviceImage& src, DeviceImage& dst);

} // namespace details
} // namespace sgm
