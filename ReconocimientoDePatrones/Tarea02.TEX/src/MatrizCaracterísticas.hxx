#ifndef __LOVDOG_MAT_CAR__
#define __LOVDOG_MAT_CAR__
#include <core.hpp>
#include <highgui.hpp>
#include <iostream>
namespace lovdog {
  void crop_from_mask(cv::Mat &src, cv::Mat &mask, cv::Mat** dst, size_t radius);
};
namespace depseek {
  int find_intersection(
      const int p1[2],
      const int p2[2],
      const int p3[2],
      const int p4[2],
      int* ix,
      int* iy
  );
};
#endif
