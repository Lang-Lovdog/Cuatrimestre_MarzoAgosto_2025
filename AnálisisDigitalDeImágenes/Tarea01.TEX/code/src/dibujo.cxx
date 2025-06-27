#include "dibujo.hxx"
#include "highgui.hpp"
#include <iostream>

// Igualación en histograma
int lovdogEqualizeIt(const cv::Mat& src, cv::Mat& dst){
  double minVal, maxVal;

  cv::minMaxLoc(src, &minVal, &maxVal);
  std::cout << "Valores [min, max] = [" << minVal <<", "<< maxVal  << "]"<< std::endl;

  cv::equalizeHist(src,dst);

  cv::minMaxLoc(dst, &minVal, &maxVal);
  std::cout << "Valores Norm [min, max] = [" << minVal <<", "<< maxVal  << "]"<< std::endl;

  return 0;
}

void showIt(cv::Mat& matrix, std::string name){
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  cv::imshow(name, matrix);
}

int GetEdge(const cv::Mat& lImage, cv::Mat& lImage_f, bool thres){
  cv::Mat lImage_o,Kernel;
  lImage.convertTo(lImage_f, CV_32F);

  Kernel = (cv::Mat_<double>(3,3) <<
      -2, -2,  0,
      -2,  0, -2,
       0, -2, -2
  );
  cv::filter2D(lImage_f, lImage_o, -1, Kernel, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
  lImage_o=cv::abs(lImage_o);
  cv::normalize(lImage_o, lImage_o,0,255,cv::NORM_MINMAX,CV_8UC1);
  if(thres)
    cv::threshold(lImage_o, lImage_f,10,255,cv::THRESH_BINARY);
  return 0;
}

