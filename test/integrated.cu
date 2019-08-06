#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <libsgm.h>
#include "internal.h"
#include "generator.hpp"

static bool is_cuda_input(sgm::EXECUTE_INOUT type) { return (type & 0x1) > 0; }
static bool is_cuda_output(sgm::EXECUTE_INOUT type) { return (type & 0x2) > 0; }

class LibSGM : public testing::TestWithParam<std::tuple<int, int, int, bool, sgm::PathType, sgm::EXECUTE_INOUT>> {
public:
	static constexpr int WIDTH = 1024, HEIGHT = 440;
	int input_depth_bits_;
	int output_depth_bits_;
	int disparity_size_;
	bool subpixel_;
	sgm::PathType path_type_;
	sgm::EXECUTE_INOUT inout_type_;

	size_t src_size_, dst_size_;
	thrust::device_vector<uint8_t> d_src_left_, d_src_right_, d_dst_;
	thrust::host_vector<uint8_t>   h_src_left_, h_src_right_, h_dst_;

	virtual void SetUp(){
		std::tie(input_depth_bits_, output_depth_bits_, disparity_size_, subpixel_, path_type_, inout_type_) = GetParam();
		src_size_ = WIDTH * HEIGHT * input_depth_bits_ / 8;
		dst_size_ = WIDTH * HEIGHT * output_depth_bits_ / 8;

		h_src_left_  = generate_random_sequence<uint8_t>(src_size_);
		h_src_right_ = generate_random_sequence<uint8_t>(src_size_);

		if(is_cuda_input(inout_type_)){
			d_src_left_ = h_src_left_;
			d_src_right_ = h_src_right_;
		}

		if(is_cuda_output(inout_type_)){
			d_dst_.resize(dst_size_);
		}else{
			h_dst_.resize(dst_size_);
		}
	}
};

TEST_P(LibSGM, Integrated){
	void *src_left, *src_right;
	if(is_cuda_input(inout_type_)){
		src_left = d_src_left_.data().get();
		src_right = d_src_right_.data().get();
	}else{
		src_left = h_src_left_.data();
		src_right = h_src_right_.data();
	}

	void *dst;
	if(is_cuda_output(inout_type_)){
		dst = d_dst_.data().get();
	}else{
		dst = h_dst_.data();
	}

	sgm::StereoSGM sgm(WIDTH, HEIGHT, disparity_size_, input_depth_bits_, output_depth_bits_, inout_type_, {10, 120, 0.95f, subpixel_, path_type_});
	sgm.execute(src_left, src_right, dst);

	CudaKernelCheck();
}

INSTANTIATE_TEST_CASE_P(LibSGM_Test_16, LibSGM, testing::Combine(
	testing::Values(8, 16),
	testing::Values(16),
	testing::Values(64, 128, 256),
	testing::Values(false, true),
	testing::Values(sgm::PathType::SCAN_4PATH, sgm::PathType::SCAN_8PATH),
	testing::Values(
		sgm::EXECUTE_INOUT_HOST2HOST,
		sgm::EXECUTE_INOUT_HOST2CUDA,
		sgm::EXECUTE_INOUT_CUDA2HOST,
		sgm::EXECUTE_INOUT_CUDA2CUDA
	)
));

INSTANTIATE_TEST_CASE_P(LibSGM_Test_8, LibSGM, testing::Combine(
	testing::Values(8, 16),
	testing::Values(8),
	testing::Values(64, 128),
	testing::Values(false),
	testing::Values(sgm::PathType::SCAN_4PATH, sgm::PathType::SCAN_8PATH),
	testing::Values(
		sgm::EXECUTE_INOUT_HOST2HOST,
		sgm::EXECUTE_INOUT_HOST2CUDA,
		sgm::EXECUTE_INOUT_CUDA2HOST,
		sgm::EXECUTE_INOUT_CUDA2CUDA
	)
));
