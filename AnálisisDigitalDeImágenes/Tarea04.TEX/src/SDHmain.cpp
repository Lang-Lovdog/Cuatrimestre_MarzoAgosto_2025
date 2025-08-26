#include "SDH_feat.hxx"
#include "OCV_Course.hxx"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char** argv){
  cv::Mat src; lovdogGetImage(argv[1], src);
  cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
  /*
  cv::Mat src = (cv::Mat_<uchar>(5,5) <<      0, 255,   0,   0,   0,
                                            255, 255, 255,   0, 255,
                                            255, 255,   0, 255, 255,
                                            255,   0, 255, 255, 255,
                                              0, 255, 255,   0, 255);
                                              */
  lovdog::SDH sdh(src);

  sdh.verbose = 1;
  uint ds[2] = {1,2} ;
  uint the_d, the_angle,
       mask =
          lovdog::SDH::HOMOGENEITY |
          lovdog::SDH::ENERGY      |
          lovdog::SDH::ENTROPY     |
          lovdog::SDH::MEAN        |
          lovdog::SDH::CONTRAST
         ;
  int the_err;
  std::string csvout =
    std::string("out/Features")+
    // Get only basename of file
    std::string(argv[1]).substr(std::string(argv[1]).find_last_of("/\\") + 1) +
    std::string(".csv");
  std::cout << "csvout: " << csvout << std::endl;

  lovdog::SDH::toCSV_WriteHeader(csvout, mask);

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
      //sdh.printFeatures(mask); std::cout <<  std::endl;
      sdh.toCSV(csvout, mask, true, argv[1]);
      the_angle++;
    }
    the_angle = 0;
    the_d++;
  }

  return 0;
}
