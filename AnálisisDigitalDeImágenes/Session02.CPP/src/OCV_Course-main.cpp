#include <iostream>
#include <highgui.hpp>
#include <core.hpp>
#include "OCV_Course.hxx"

int main (int argc, char* argv[]){
  cv::Mat Imagen, Mask;
  lovdogGetImage("Journey.jpg", Imagen);
  showIt(Imagen, "Journey");
  cv::cvtColor(Imagen, Imagen, cv::COLOR_BGR2HLS);
  showIt(Imagen, "Journey HLS");
  cv::waitKey(0);
  Mask = cv::Mat::zeros(Imagen.size(), CV_8UC1);
  return 0;
}
