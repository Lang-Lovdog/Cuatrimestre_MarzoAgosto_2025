#ifndef __LOVDOG_SDH_FEAT_HXX__
#define __LOVDOG_SDH_FEAT_HXX__
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <vector>

namespace lovdog{

  class SDH {
    public:
      SDH();
      SDH(const SDH& sdh);
      SDH(size_t d, size_t angle);
      static const int
        ANGLE_0=0,
        ANGLE_45=1,
        ANGLE_90=2,
        ANGLE_135=3
      ;
      static const int
        DIFF=0,
        SUM=1
      ;
      int
        d,
        angle,
        dx,
        dy
      ;
      double
        mean[2],
        variance[2],
        correlation,
        contrast,
        homogeneity,
        shadowness,
        prominence
      ;
      double* at(const int which_one, int index);
      double* atRel(const int which_one, size_t index);
      static int getSDH(const cv::Mat& src, SDH& sdh);
      static void computeFeatures(SDH& sdh);
      static void toCSV(std::vector<SDH>& sdh, std::string filename);

    private:
      double
        sumHist[511],
        diffHist[511]
      ;
  };
  typedef std::vector<SDH> SDHs;


};

#endif
