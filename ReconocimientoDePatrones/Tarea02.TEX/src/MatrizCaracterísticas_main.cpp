#include <core.hpp>
#include <highgui.hpp>
#include "MatrizCaracterísticas.hxx"

int main(int argv,char** argc){
  if(argv != 3){
    std::cerr << "Usage: " << argc[0] << " <image> <mask>" << std::endl;
    return -1;
  }
  cv::Mat src = cv::imread(argc[1], CV_8UC1);
  cv::Mat mask = cv::imread(argc[2], CV_8UC1);
  cv::Mat *dst;
  lovdog::crop_from_mask(src,mask,&dst,10);
  cv::imshow("Crop 1 TL",dst[0]);
  cv::imshow("Crop 2 TR",dst[1]);
  cv::imshow("Crop 3 BL",dst[2]);
  cv::imshow("Crop 4 BR",dst[3]);
  cv::imshow("Crop 5 CT",dst[4]);
  cv::waitKey(0);
  return 0;
}
