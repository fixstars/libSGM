#include <pybind11/pybind11.h>

#include <libsgm.h>
#include <libsgm_wrapper.h>

#ifdef BUILD_OPENCV_WRAPPER

#include <pybind11/numpy.h>

namespace pybind11 {
namespace detail {
template<>
struct type_caster<cv::Mat> {
 public:
 PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  //! 1. cast numpy.ndarray to cv::Mat
  bool load(handle obj, bool) {
    array b = reinterpret_borrow<array>(obj);
    buffer_info info = b.request();

    //const int ndims = (int)info.ndim;
    int nh = 1;
    int nw = 1;
    int nc = 1;
    int ndims = info.ndim;
    if (ndims == 2) {
      nh = info.shape[0];
      nw = info.shape[1];
    } else if (ndims == 3) {
      nh = info.shape[0];
      nw = info.shape[1];
      nc = info.shape[2];
    } else {
      char msg[64];
      std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
      throw std::logic_error(msg);
      return false;
    }

    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8UC(nc);
    } else if (info.format == format_descriptor<unsigned short>::format()) {
      dtype = CV_16UC(nc);
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32SC(nc);
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32FC(nc);
    } else {
      throw std::logic_error("Unsupported type, only support uchar, int32, float");
      return false;
    }

    value = cv::Mat(nh, nw, dtype, info.ptr);
    return true;
  }

  //! 2. cast cv::Mat to numpy.ndarray
  static handle cast(const cv::Mat &mat, return_value_policy, handle defval) {
//    UNUSED(defval);


    std::string format = format_descriptor<unsigned char>::format();
    size_t elemsize = sizeof(unsigned char);
    int nw = mat.cols;
    int nh = mat.rows;
    int nc = mat.channels();
    int depth = mat.depth();
    int type = mat.type();
    int dim = (depth == type) ? 2 : 3;

    if (depth == CV_8U) {
      format = format_descriptor<unsigned char>::format();
      elemsize = sizeof(unsigned char);
    } else if (depth == CV_16U) {
      format = format_descriptor<unsigned short>::format();
      elemsize = sizeof(unsigned short);
    } else if (depth == CV_16S) {
      format = format_descriptor<short>::format();
      elemsize = sizeof(short);
    } else if (depth == CV_32S) {
      format = format_descriptor<int>::format();
      elemsize = sizeof(int);
    } else if (depth == CV_32F) {
      format = format_descriptor<float>::format();
      elemsize = sizeof(float);
    } else {
      throw std::logic_error("Unsupport type!");
    }

    std::vector<size_t> bufferdim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferdim = {(size_t) nh, (size_t) nw};
      strides = {elemsize * (size_t) nw, elemsize};
    } else if (dim == 3) {
      bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
      strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
    }
    return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
  }
};
}
}//! end namespace pybind11::detail

#endif // BUILD_OPENCV_WRAPPER


#define RW(type_name, field_name) .def_readwrite(#field_name, &type_name::field_name)
namespace py = pybind11;

PYBIND11_MODULE(pysgm, m) {

  py::enum_<sgm::EXECUTE_INOUT>(m, "EXECUTE_INOUT")
      .value("EXECUTE_INOUT_HOST2HOST", sgm::EXECUTE_INOUT::EXECUTE_INOUT_HOST2HOST)
      .value("EXECUTE_INOUT_HOST2CUDA", sgm::EXECUTE_INOUT::EXECUTE_INOUT_HOST2CUDA)
      .value("EXECUTE_INOUT_CUDA2HOST", sgm::EXECUTE_INOUT::EXECUTE_INOUT_CUDA2HOST)
      .value("EXECUTE_INOUT_CUDA2CUDA", sgm::EXECUTE_INOUT::EXECUTE_INOUT_CUDA2CUDA)
      ;

  py::enum_<sgm::PathType>(m, "PathType")
      .value("SCAN_4PATH", sgm::PathType::SCAN_4PATH)
      .value("SCAN_8PATH", sgm::PathType::SCAN_8PATH)
      ;

  py::class_<sgm::StereoSGM> StereoSGM(m, "StereoSGM");

  py::class_<sgm::StereoSGM::Parameters>(StereoSGM, "Parameters")
      .def(py::init<int, int, float, bool, sgm::PathType, int, int>(),
           py::arg("P1") = 10,
           py::arg("P2") = 120,
           py::arg("uniqueness") = 0.95f,
           py::arg("subpixel") = false,
           py::arg("PathType") = sgm::PathType::SCAN_8PATH,
           py::arg("min_disp") = 0,
           py::arg("LR_max_diff") = 1
      )
          RW(sgm::StereoSGM::Parameters, P1)
          RW(sgm::StereoSGM::Parameters, P2)
          RW(sgm::StereoSGM::Parameters, uniqueness)
          RW(sgm::StereoSGM::Parameters, subpixel)
          RW(sgm::StereoSGM::Parameters, path_type)
          RW(sgm::StereoSGM::Parameters, min_disp)
          RW(sgm::StereoSGM::Parameters, LR_max_diff);

  StereoSGM
      .def(py::init<int, int, int, int, int, sgm::EXECUTE_INOUT, const sgm::StereoSGM::Parameters &>(),
           py::arg("width") = 612,
           py::arg("height") = 514,
           py::arg("disparity_size") = 128,
           py::arg("input_depth_bits") = 8U,
           py::arg("output_depth_bits") = 8U,
           py::arg("inout_type") = sgm::EXECUTE_INOUT::EXECUTE_INOUT_HOST2HOST,
           py::arg("param") = sgm::StereoSGM::Parameters()
               )
      .def(py::init<int, int, int, int, int, int, int, sgm::EXECUTE_INOUT, const sgm::StereoSGM::Parameters &>())
      .def("execute", [](sgm::StereoSGM &w, uintptr_t left_pixels, uintptr_t right_pixels, uintptr_t dst) {
        w.execute((void *)left_pixels, (void *)right_pixels, (void *)dst);
      })
      .def("get_invalid_disparity", &sgm::StereoSGM::get_invalid_disparity)
      ;

#ifdef BUILD_OPENCV_WRAPPER

  py::class_<sgm::LibSGMWrapper> LibSGMWrapper(m, "LibSGMWrapper");

  LibSGMWrapper
      .def(py::init<int, int, int, float, bool, sgm::PathType, int, int>(),
           py::arg("numDisparity") = 128,
           py::arg("P1") = 10,
           py::arg("P2") = 120,
           py::arg("uniquenessRatio") = 0.95f,
           py::arg("subpixel") = false,
           py::arg("pathType") = sgm::PathType::SCAN_8PATH,
           py::arg("minDisparity") = 0,
           py::arg("lrMaxDiff") = 1)
      .def("getInvalidDisparity", &sgm::LibSGMWrapper::getInvalidDisparity)
      .def("hasSubpixel", &sgm::LibSGMWrapper::hasSubpixel)
      .def("execute", [](sgm::LibSGMWrapper &w, cv::Mat &left_pixels, const cv::Mat &right_pixels) {
        cv::Mat disp;
        w.execute(left_pixels, right_pixels, disp);
        return disp;
      });

#endif // BUILD_OPENCV_WRAPPER

  m.def("SUBPIXEL_SCALE", []() { return sgm::StereoSGM::SUBPIXEL_SCALE; });
  m.def("SUBPIXEL_SHIFT", []() { return sgm::StereoSGM::SUBPIXEL_SHIFT; });
}
