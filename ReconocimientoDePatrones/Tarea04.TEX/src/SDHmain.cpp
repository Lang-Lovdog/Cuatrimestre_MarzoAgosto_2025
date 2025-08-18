#include "SDH_feat.hxx"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv){
  lovdog::SDHs sdhs;
  cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  for(size_t delta=3; delta<26; delta+=2)
    sdhs.push_back(lovdog::SDH(delta, lovdog::SDH::ANGLE_0));
  return 0;
}
