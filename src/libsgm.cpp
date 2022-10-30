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

#include <iostream>

#include <libsgm.h>

#include "internal.h"
#include "device_image.h"

namespace sgm {
	static bool is_cuda_input(EXECUTE_INOUT type) { return (type & 0x1) > 0; }
	static bool is_cuda_output(EXECUTE_INOUT type) { return (type & 0x2) > 0; }

	static bool has_enough_depth(int output_depth_bits, int disparity_size, int min_disp, bool subpixel)
	{
		// simulate minimum/maximum value
		int64_t max = static_cast<int64_t>(disparity_size) + min_disp - 1;
		if (subpixel) {
			max *= sgm::StereoSGM::SUBPIXEL_SCALE;
			max += sgm::StereoSGM::SUBPIXEL_SCALE - 1;
		}

		if (1ll << output_depth_bits <= max)
			return false;

		if (min_disp <= 0) {
			// whether or not output can be represented by signed
			int64_t min = static_cast<int64_t>(min_disp) - 1;
			if (subpixel) {
				min *= sgm::StereoSGM::SUBPIXEL_SCALE;
			}

			if (min < -(1ll << (output_depth_bits - 1))
			    || 1ll << (output_depth_bits - 1) <= max)
				return false;
		}

		return true;
	}

	class StereoSGM::Impl
	{
	public:

		Impl(int width, int height, int disparity_size, int input_depth_bits, int output_depth_bits, int src_pitch, int dst_pitch,
			EXECUTE_INOUT inout_type, const Parameters& param) :
			width_(width),
			height_(height),
			disparity_size_(disparity_size),
			input_depth_bits_(input_depth_bits),
			output_depth_bits_(output_depth_bits),
			src_pitch_(src_pitch),
			dst_pitch_(dst_pitch),
			inout_type_(inout_type),
			param_(param)
		{
			// check values
			if (input_depth_bits_ != 8 && input_depth_bits_ != 16 && output_depth_bits_ != 8 && output_depth_bits_ != 16) {
				width_ = height_ = input_depth_bits_ = output_depth_bits_ = disparity_size_ = 0;
				throw std::logic_error("depth bits must be 8 or 16");
			}
			if (disparity_size_ != 64 && disparity_size_ != 128 && disparity_size != 256) {
				width_ = height_ = input_depth_bits_ = output_depth_bits_ = disparity_size_ = 0;
				throw std::logic_error("disparity size must be 64, 128 or 256");
			}
			if (!has_enough_depth(output_depth_bits, disparity_size, param_.min_disp, param_.subpixel)) {
				width_ = height_ = input_depth_bits_ = output_depth_bits_ = disparity_size_ = 0;
				throw std::logic_error("output depth bits must be sufficient for representing output value");
			}
			if (param_.path_type != PathType::SCAN_4PATH && param_.path_type != PathType::SCAN_8PATH) {
				width_ = height_ = input_depth_bits_ = output_depth_bits_ = disparity_size_ = 0;
				throw std::logic_error("Path type must be PathType::SCAN_4PATH or PathType::SCAN_8PATH");
			}

			src_type_ = input_depth_bits == 8 ? SGM_8U : SGM_16U;
			dst_type_ = output_depth_bits == 8 ? SGM_8U : SGM_16U;

			if (!is_cuda_input(inout_type_)) {
				d_src_left_.create(height, width, src_type_, src_pitch);
				d_src_right_.create(height, width, src_type_, src_pitch);
			}

			d_left_disp_.create(height, width, SGM_16U, dst_pitch);
			d_right_disp_.create(height, width, SGM_16U, dst_pitch);

			d_tmp_left_disp_.create(height, width, SGM_16U, dst_pitch);
			d_tmp_right_disp_.create(height, width, SGM_16U, dst_pitch);

			d_census_left_.create(height, width, SGM_32U);
			d_census_right_.create(height, width, SGM_32U);
			d_census_left_.fill_zero();
			d_census_right_.fill_zero();

			const int num_paths = param.path_type == PathType::SCAN_4PATH ? 4 : 8;
			d_cost_.create(num_paths, width * height * disparity_size, SGM_8U);
		}

		~Impl() {
		}

		void execute(const void* left_pixels, const void* right_pixels, void* dst) {

			const void *d_input_left, *d_input_right;

			if (is_cuda_input(inout_type_)) {
				d_input_left = left_pixels;
				d_input_right = right_pixels;
			}
			else {
				d_src_left_.upload(left_pixels);
				d_src_right_.upload(right_pixels);
				d_input_left = d_src_left_.data;
				d_input_right = d_src_right_.data;
			}

			void* d_tmp_left_disp = d_tmp_left_disp_.data;
			void* d_tmp_right_disp = d_tmp_right_disp_.data;
			void* d_left_disp = d_left_disp_.data;
			void* d_right_disp = d_right_disp_.data;
			uint32_t* d_census_left = (uint32_t*)d_census_left_.data;
			uint32_t* d_census_right = (uint32_t*)d_census_right_.data;
			cost_type* d_cost = (cost_type*)d_cost_.data;

			if (is_cuda_output(inout_type_) && output_depth_bits_ == 16)
				d_left_disp = dst; // when threre is no device-host copy or type conversion, use passed buffer

			sgm::details::census_transform(d_input_left, d_census_left, width_, height_, src_pitch_, input_depth_bits_);
			sgm::details::census_transform(d_input_right, d_census_right, width_, height_, src_pitch_, input_depth_bits_);
			sgm::details::cost_aggregation(d_census_left, d_census_right, d_cost, width_, height_, disparity_size_, param_.P1, param_.P2, param_.path_type, param_.min_disp);
			sgm::details::winner_takes_all(d_cost, (uint16_t*)d_tmp_left_disp, (uint16_t*)d_tmp_right_disp, width_, height_, dst_pitch_,
				disparity_size_, param_.uniqueness, param_.subpixel, param_.path_type);

			sgm::details::median_filter((uint16_t*)d_tmp_left_disp, (uint16_t*)d_left_disp, width_, height_, dst_pitch_);
			sgm::details::median_filter((uint16_t*)d_tmp_right_disp, (uint16_t*)d_right_disp, width_, height_, dst_pitch_);
			sgm::details::check_consistency((uint16_t*)d_left_disp, (uint16_t*)d_right_disp, d_input_left, width_, height_, input_depth_bits_, src_pitch_, dst_pitch_, param_.subpixel, param_.LR_max_diff);
			sgm::details::correct_disparity_range((uint16_t*)d_left_disp, width_, height_, dst_pitch_, param_.subpixel, param_.min_disp);

			if (!is_cuda_output(inout_type_) && output_depth_bits_ == 8) {
				sgm::details::cast_16bit_8bit_array((const uint16_t*)d_left_disp, (uint8_t*)d_tmp_left_disp, dst_pitch_ * height_);
				CudaSafeCall(cudaMemcpy(dst, d_tmp_left_disp, sizeof(uint8_t) * dst_pitch_ * height_, cudaMemcpyDeviceToHost));
			}
			else if (is_cuda_output(inout_type_) && output_depth_bits_ == 8) {
				sgm::details::cast_16bit_8bit_array((const uint16_t*)d_left_disp, (uint8_t*)dst, dst_pitch_ * height_);
			}
			else if (!is_cuda_output(inout_type_) && output_depth_bits_ == 16) {
				CudaSafeCall(cudaMemcpy(dst, d_left_disp, sizeof(uint16_t) * dst_pitch_ * height_, cudaMemcpyDeviceToHost));
			}
			else if (is_cuda_output(inout_type_) && output_depth_bits_ == 16) {
				// optimize! no-copy!
			}
			else {
				std::cerr << "not impl" << std::endl;
			}
		}

		int get_invalid_disparity() const {
			return (param_.min_disp - 1) * (param_.subpixel ? SUBPIXEL_SCALE : 1);
		}

	private:

		int width_;
		int height_;
		int disparity_size_;
		int input_depth_bits_;
		int output_depth_bits_;
		int src_pitch_;
		int dst_pitch_;
		EXECUTE_INOUT inout_type_;
		Parameters param_;
		ImageType src_type_, dst_type_;

		DeviceImage d_src_left_;
		DeviceImage d_src_right_;
		DeviceImage d_left_disp_;
		DeviceImage d_right_disp_;
		DeviceImage d_tmp_left_disp_;
		DeviceImage d_tmp_right_disp_;
		DeviceImage d_census_left_;
		DeviceImage d_census_right_;
		DeviceImage d_cost_;
	};

	StereoSGM::StereoSGM(int width, int height, int disparity_size, int src_depth, int dst_depth,
		EXECUTE_INOUT inout_type, const Parameters& param)
	{
		impl_ = new Impl(width, height, disparity_size, src_depth, dst_depth, width, width, inout_type, param);
	}

	StereoSGM::StereoSGM(int width, int height, int disparity_size, int src_depth, int dst_depth, int src_pitch, int dst_pitch,
		EXECUTE_INOUT inout_type, const Parameters& param)
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
}
