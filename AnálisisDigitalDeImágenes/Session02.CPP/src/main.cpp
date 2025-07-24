#include <iostream>
#include <highgui.hpp>
#include <core.hpp>
#include "OCV_Course.hxx"
#include "core/hal/interface.h"
#include "core/matx.hpp"
#include "dibujo.hxx"
#include "imgproc.hpp"

bool lovdogPxIsSkin(const cv::Vec3b& px);

int main (int argc, char* argv[]){
  cv::Mat Imagen, Mask;
  lovdogGetImage("Journey.jpg", Imagen);
  Mask = cv::Mat::zeros(Imagen.rows,Imagen.cols,CV_8UC1);

  showIt(Imagen, "Journey");
  cv::cvtColor(Imagen, Imagen, cv::COLOR_BGR2HLS);

  for(int r=0; r < Imagen.rows; ++r)
    for(int c=0; c < Imagen.cols; ++c){
      cv::Vec3b px = Imagen.at<cv::Vec3b>(r,c);
      if(lovdogPxIsSkin(px))
        Mask.at<uchar>(r,c) = 255;
    }
  cv::cvtColor(Imagen, Imagen, cv::COLOR_HLS2BGR);

  for(int r=0; r < Imagen.rows; ++r)
    for(int c=0; c < Imagen.cols; ++c){
      if(!Mask.at<uchar>(r,c))
        Imagen.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,0);
    }

  showIt(Mask,"Mask");
  showIt(Imagen,"Journey After Mask");

  cv::waitKey(0);
  return 0;
}

bool lovdogPxIsSkin(const cv::Vec3b& px){
  // s <= 50
  // 0.5 <= l/s <= 3.0
  // 165 <= H <= 14
  double ratio_ls = (double)px[1]/px[2];
  return
    ( px[2] > 49     ) &&
    ( 0.5 < ratio_ls ) &&
    ( ratio_ls < 3.0 ) &&
    ( px[0] < 15   ||  px[0] > 164 )
  ;
}
