#ifndef __LOVDOG_COLOR_TRANSFER__
#define __LOVDOG_COLOR_TRANSFER__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include "OCV_Course.hxx"


namespace lovdog {
  const float _Matrix_RGB2LMS[9] = {
    0.3811, 0.5783, 0.0402,
    0.1967, 0.7244, 0.0782,
    0.0241, 0.1288, 0.8444
  };
  const float _Matrix_LMS2RGB[9] = {
    4.4679, -3.5873, 0.1193,
   -1.2186, 2.3809, -0.1624,
    0.0497, -0.2439, 1.2045
    
  };
  const float _Matrix_LMS2lAphaBeta[3] = {
    (float)cv::sqrt(1.0/3),
    (float)cv::sqrt(1.0/6),
    (float)cv::sqrt(1.0/2)
  };
  const float _Matrix_lAphaBeta2LMS[3] = {
    (float)cv::sqrt(3)/3,
    (float)cv::sqrt(6)/6,
    (float)cv::sqrt(2)/2
  };
  const float Matrix_lAlphaBetaLMSK[9] = {
    1, 1, 1,
    1, 1,-2,
    1,-1, 0
  };
  
  // Create OpenCV matrices
  const cv::Mat RGB2LMS = cv::Mat(3, 3, CV_32FC1, (void*)_Matrix_RGB2LMS).clone();
  const cv::Mat LMS2RGB = cv::Mat(3, 3, CV_32FC1, (void*)_Matrix_LMS2RGB).clone();
  const cv::Mat LMS2LAB_weights = cv::Mat(1, 3, CV_32FC1, (void*)_Matrix_LMS2lAphaBeta).clone();

  const uint LMS2BGR = 0;
  const uint BGR2LMS = 1;
  const uint LMS2lAlphaBeta = 2;
  const uint lAlphaBeta2LMS = 3;

  void cvtColor(const cv::Mat& input, cv::Mat& output, uint code);
  void colorTransfer(const cv::Mat& input, const cv::Mat& reference, cv::Mat& output);
  int mainColorTransfer(int argc, char** argv);

};


#endif
