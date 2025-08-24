#include "SDH_feat.hxx"
#include "OCV_Course.hxx"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv){
  cv::Mat src; lovdogGetImage(argv[1], src);
  cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
  lovdog::SDH sdh(src);

  bool header = true;
  uint ds[2] = {1,2} ;
  uint the_d, the_angle;
  int the_err;

  lovdog::SDH::toCSV_WriteHeader("Fearures.csv", lovdog::SDH::ALL);
  the_d = the_angle = 0;
  while(the_d < 2){
    while(the_angle < 4){
      if((the_err=sdh.getSDH(ds[the_d], lovdog::SDH::ANGLE[the_angle]))){
        std::cout << "Error in getSDH, err: " << the_err << std::endl;
        ++the_angle;
        continue;
      }
      sdh.computeFeatures();
      // Print all features
      sdh.printFeatures();
      sdh.toCSV("Fearures.csv", lovdog::SDH::ALL, true, argv[1]);
      the_angle++;
    }
    the_angle = 0;
    the_d++;
  }

  return 0;
}
