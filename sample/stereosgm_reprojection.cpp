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
#include <iomanip>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

// Camera Parameters
struct CameraParameters
{
	float fu;                 //!< focal length x (pixel)
	float fv;                 //!< focal length y (pixel)
	float u0;                 //!< principal point x (pixel)
	float v0;                 //!< principal point y (pixel)
	float baseline;           //!< baseline (meter)
	float height;             //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
	float tilt;               //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO
};

// Transformation between pixel coordinate and world coordinate
struct CoordinateTransform
{
	CoordinateTransform(const CameraParameters& camera) : camera(camera)
	{
		sinTilt = (sinf(camera.tilt));
		cosTilt = (cosf(camera.tilt));
		bf = camera.baseline * camera.fu;
		invfu = 1.f / camera.fu;
		invfv = 1.f / camera.fv;
	}

	inline cv::Point3f imageToWorld(const cv::Point2f& pt, float d) const
	{
		const float u = pt.x;
		const float v = pt.y;

		const float Zc = bf / d;
		const float Xc = invfu * (u - camera.u0) * Zc;
		const float Yc = invfv * (v - camera.v0) * Zc;

		const float Xw = Xc;
		const float Yw = Yc * cosTilt + Zc * sinTilt;
		const float Zw = Zc * cosTilt - Yc * sinTilt;

		return cv::Point3f(Xw, Yw, Zw);
	}

	CameraParameters camera;
	float sinTilt, cosTilt, bf, invfu, invfv;
};

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

void reprojectPointsTo3D(const cv::Mat& disparity, const CameraParameters& camera, std::vector<cv::Point3f>& points, bool subpixeled)
{
	CV_Assert(disparity.type() == CV_32F);

	CoordinateTransform tf(camera);

	points.clear();
	points.reserve(disparity.rows * disparity.cols);

	for (int y = 0; y < disparity.rows; y++)
	{
		for (int x = 0; x < disparity.cols; x++)
		{
			const float d = disparity.at<float>(y, x);
			if (d > 0)
				points.push_back(tf.imageToWorld(cv::Point(x, y), d));
		}
	}
}

void drawPoints3D(const std::vector<cv::Point3f>& points, cv::Mat& draw)
{
	const int SIZE_X = 512;
	const int SIZE_Z = 1024;
	const int maxz = 20; // [meter]
	const double pixelsPerMeter = 1. * SIZE_Z / maxz;

	draw = cv::Mat::zeros(SIZE_Z, SIZE_X, CV_8UC3);

	for (const cv::Point3f& pt : points)
	{
		const float X = pt.x;
		const float Z = pt.z;

		const int u = cvRound(pixelsPerMeter * X) + SIZE_X / 2;
		const int v = SIZE_Z - cvRound(pixelsPerMeter * Z);

		const cv::Scalar color = computeColor(std::min(Z, 1.f * maxz) / maxz);
		cv::circle(draw, cv::Point(u, v), 1, color);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 4) {
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml [disp_size] [subpixel_enable(0: false, 1:true)]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int first_frame = 1;

	cv::Mat I1 = cv::imread(format_string(argv[1], first_frame), -1);
	cv::Mat I2 = cv::imread(format_string(argv[2], first_frame), -1);
	const cv::FileStorage fs(argv[3], cv::FileStorage::READ);
	const int disp_size = argc >= 5 ? std::stoi(argv[4]) : 128;
	const bool subpixel = argc >= 6 ? std::stoi(argv[5]) != 0 : true;
	const int output_depth = 16;

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(fs.isOpened(), "camera.xml read failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

	// read camera parameters
	CameraParameters camera;
	camera.fu = fs["FocalLengthX"];
	camera.fv = fs["FocalLengthY"];
	camera.u0 = fs["CenterX"];
	camera.v0 = fs["CenterY"];
	camera.baseline = fs["BaseLine"];
	camera.tilt = fs["Tilt"];

	const int width = I1.cols;
	const int height = I1.rows;

	const int input_depth = I1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int output_bytes = output_depth * width * height / 8;

	const sgm::StereoSGM::Parameters params{10, 120, 0.95f, subpixel};

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, params);

	cv::Mat disparity(height, width, CV_16S);
	cv::Mat disparity_8u, disparity_32f, disparity_color, draw;
	std::vector<cv::Point3f> points;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

	for (int frame_no = first_frame;; frame_no++) {

		I1 = cv::imread(format_string(argv[1], frame_no), -1);
		I2 = cv::imread(format_string(argv[2], frame_no), -1);
		if (I1.empty() || I2.empty()) {
			frame_no = first_frame;
			continue;
		}

		cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		// draw results
		if (I1.type() != CV_8U) {
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
		}

		disparity.convertTo(disparity_32f, CV_32F, subpixel ? 1. / sgm::StereoSGM::SUBPIXEL_SCALE : 1);
		reprojectPointsTo3D(disparity_32f, camera, points, subpixel);
		drawPoints3D(points, draw);

		disparity_32f.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		disparity_color.setTo(cv::Scalar(0, 0, 0), disparity_32f < 0); // invalid disparity will be negative
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("left image", I1);
		cv::imshow("disparity", disparity_color);
		cv::imshow("points", draw);

		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}

	return 0;
}
