#include "MatrizCaracteristicas.hxx"
#include <iostream>

int main(int argv,char** argc){
  if(argv != 3){
    std::cerr << "Usage: " << argc[0] << " <image> <mask>" << std::endl;
    return -1;
  }
  std::string filename(argc[1]), extension;
  filename = filename.substr(0, filename.find_last_of("."));
  extension = ".png";
  lovdog::Gallery crops;
  lovdog::Contours contours;
  cv::Mat src = cv::imread(argc[1], CV_8UC1);
  cv::Mat mask = cv::imread(argc[2], CV_8UC1);
  lovdog::getMaskContours(mask, contours);
  lovdog::cropFromMaskContours(src, contours, crops, 15, 5);

  lovdog::showGallery(crops, filename+"_crop-");
  lovdog::exportGallery(crops, filename+"_crop-", "", extension);

  cv::waitKey(1400);
  return 0;
}
