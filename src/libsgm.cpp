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

#include <libsgm.h>

#include <iostream>

#include "internal.h"
#include "host_utility.h"

namespace sgm
{

static bool has_enough_depth(int dst_depth, int disparity_size, int min_disp, bool subpixel)
{
	// simulate minimum/maximum value
	int64_t max = static_cast<int64_t>(disparity_size) + min_disp - 1;
	if (subpixel) {
		max *= sgm::StereoSGM::SUBPIXEL_SCALE;
		max += sgm::StereoSGM::SUBPIXEL_SCALE - 1;
	}

	if (1ll << dst_depth <= max)
		return false;

	if (min_disp <= 0) {
		// whether or not output can be represented by signed
		int64_t min = static_cast<int64_t>(min_disp) - 1;
		if (subpixel) {
			min *= sgm::StereoSGM::SUBPIXEL_SCALE;
		}

		if (min < -(1ll << (dst_depth - 1))
			|| 1ll << (dst_depth - 1) <= max)
			return false;
	}

	return true;
}

class StereoSGM::Impl
{
public:

	Impl(int width, int height, int disparity_size, int src_depth, int dst_depth, int src_pitch, int dst_pitch,
		ExecuteInOut inout_type, const Parameters& param) :
		width_(width),
		height_(height),
		disp_size_(disparity_size),
		src_pitch_(src_pitch),
		dst_pitch_(dst_pitch),
		param_(param)
	{
		// check values
		SGM_ASSERT(src_depth == 8 || src_depth == 16, "src depth bits must be 8 or 16");
		SGM_ASSERT(dst_depth == 8 || dst_depth == 16, "dst depth bits must be 8 or 16");
		SGM_ASSERT(disparity_size == 64 || disparity_size == 128 || disparity_size == 256, "disparity size must be 64 or 128 or 256");
		SGM_ASSERT(has_enough_depth(dst_depth, disparity_size, param_.min_disp, param_.subpixel),
			"output depth bits must be sufficient for representing output value");

		src_type_ = src_depth == 8 ? SGM_8U : SGM_16U;
		dst_type_ = dst_depth == 8 ? SGM_8U : SGM_16U;

		is_src_devptr_ = (inout_type & 0x01) > 0;
		is_dst_devptr_ = (inout_type & 0x02) > 0;

		if (!is_src_devptr_) {
			d_srcL_.create(height, width, src_type_, src_pitch);
			d_srcR_.create(height, width, src_type_, src_pitch);
		}

		d_censusL_.create(height, width, SGM_32U);
		d_censusR_.create(height, width, SGM_32U);
		d_censusL_.fill_zero();
		d_censusR_.fill_zero();

		d_tmpL_.create(height, width, SGM_16U, dst_pitch);
		d_tmpR_.create(height, width, SGM_16U, dst_pitch);

		if (!(is_dst_devptr_ && dst_type_ == SGM_16U)) {
			d_dispL_.create(height, width, SGM_16U, dst_pitch);
		}
		d_dispR_.create(height, width, SGM_16U, dst_pitch);
	}

	void execute(const void* srcL, const void* srcR, void* dst)
	{
		if (is_src_devptr_) {
			d_srcL_.create((void*)srcL, height_, width_, src_type_, src_pitch_);
			d_srcR_.create((void*)srcR, height_, width_, src_type_, src_pitch_);
		}
		else {
			d_srcL_.upload(srcL);
			d_srcR_.upload(srcR);
		}
		if (is_dst_devptr_ && dst_type_ == SGM_16U) {
			// when threre is no device-host copy or type conversion, use passed buffer
			d_dispL_.create((void*)dst, height_, width_, SGM_16U, dst_pitch_);
		}

		// census transform
		details::census_transform(d_srcL_, d_censusL_);
		details::census_transform(d_srcR_, d_censusR_);

		// cost aggregation
		details::cost_aggregation(d_censusL_, d_censusR_, d_cost_, disp_size_,
			param_.P1, param_.P2, param_.path_type, param_.min_disp);

		// winner-takes-all
		details::winner_takes_all(d_cost_, d_tmpL_, d_tmpR_, disp_size_,
			param_.uniqueness, param_.subpixel, param_.path_type);

		// post filtering
		details::median_filter(d_tmpL_, d_dispL_);
		details::median_filter(d_tmpR_, d_dispR_);

		// consistency check
		details::check_consistency(d_dispL_, d_dispR_, d_srcL_, param_.subpixel, param_.LR_max_diff);
		details::correct_disparity_range(d_dispL_, param_.subpixel, param_.min_disp);

		if (!is_dst_devptr_ && dst_type_ == SGM_8U) {
			details::cast_16bit_to_8bit(d_dispL_, d_tmpL_);
			d_tmpL_.download(dst);
		}
		else if (is_dst_devptr_ && dst_type_ == SGM_8U) {
			DeviceImage d_dst(dst, height_, width_, SGM_8U, dst_pitch_);
			details::cast_16bit_to_8bit(d_dispL_, d_dst);
		}
		else if (!is_dst_devptr_ && dst_type_ == SGM_16U) {
			d_dispL_.download(dst);
		}
		else if (is_dst_devptr_ && dst_type_ == SGM_16U) {
			// optimize! no-copy!
		}
		else {
			std::cerr << "not impl" << std::endl;
		}
	}

	int get_invalid_disparity() const
	{
		return (param_.min_disp - 1) * (param_.subpixel ? SUBPIXEL_SCALE : 1);
	}

private:

	int width_;
	int height_;
	int disp_size_;
	int src_pitch_;
	int dst_pitch_;
	Parameters param_;

	ImageType src_type_;
	ImageType dst_type_;
	bool is_src_devptr_;
	bool is_dst_devptr_;

	DeviceImage d_srcL_;
	DeviceImage d_srcR_;
	DeviceImage d_censusL_;
	DeviceImage d_censusR_;
	DeviceImage d_cost_;
	DeviceImage d_tmpL_;
	DeviceImage d_tmpR_;
	DeviceImage d_dispL_;
	DeviceImage d_dispR_;
};

StereoSGM::StereoSGM(int width, int height, int disparity_size, int src_depth, int dst_depth,
	ExecuteInOut inout_type, const Parameters& param)
{
	impl_ = new Impl(width, height, disparity_size, src_depth, dst_depth, width, width, inout_type, param);
}

StereoSGM::StereoSGM(int width, int height, int disparity_size, int src_depth, int dst_depth, int src_pitch, int dst_pitch,
	ExecuteInOut inout_type, const Parameters& param)
{
	impl_ = new Impl(width, height, disparity_size, src_depth, dst_depth, src_pitch, dst_pitch, inout_type, param);
}

StereoSGM::~StereoSGM()
{
	delete impl_;
}

void StereoSGM::execute(const void* srcL, const void* srcR, void* dst)
{
	impl_->execute(srcL, srcR, dst);
}

int StereoSGM::get_invalid_disparity() const
{
	return impl_->get_invalid_disparity();
}

} // namespace sgm
