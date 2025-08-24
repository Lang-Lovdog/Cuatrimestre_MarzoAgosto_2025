#ifndef __LOVDOG_SDH_FEAT_HXX__
#define __LOVDOG_SDH_FEAT_HXX__
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace lovdog{

  class SDH {
    public:
      SDH();
      SDH(const SDH& sdh);
      SDH(const cv::Mat& src);
      SDH(size_t d, size_t angle);
      SDH(const cv::Mat& src, uint d, uint angle);
      static const uint
        ANGLE_0=0,
        ANGLE_45=45,
        ANGLE_90=90,
        ANGLE_135=135,
        ANGLE_180=180
      ;
      static constexpr uint ANGLE[5] = {ANGLE_0, ANGLE_45, ANGLE_90, ANGLE_135, ANGLE_180};
      static const int
        DIFF=0,
        SUM=1
      ;
      static const uint
        ALL            = ~0,
        _MEAN_DIFF     = 0b000000000001,
        _MEAN_SUM      = 0b000000000010,
        _VARIANCE_DIFF = 0b000000000100,
        _VARIANCE_SUM  = 0b000000001000,
        CORRELATION    = 0b000000010000,
        CONTRAST       = 0b000000100000,
        HOMOGENEITY    = 0b000001000000,
        SHADOWNESS     = 0b000010000000,
        PROMINENCE     = 0b000100000000,
        ENERGY         = 0b001000000000,
        ENTROPY        = 0b010000000000,
        MEAN           = 0b100000000000
      ;
      cv::Mat src;
      uint
        d,
        angle
      ;
      int
        dx,
        dy
      ;
      double
        _mean[2],
        _variance[2],
        correlation,
        contrast,
        homogeneity,
        shadowness,
        prominence,
        energy,
        entropy,
        mean
      ;
      void set(int d, uint angle);
      double* at(const int which_one, int index);
      double* atRel(const int which_one, size_t index);
      void toCSV(std::string filename, unsigned int HEADER=0, bool append=false, std::string name="SDH");
      int getSDH(void);
      int getSDH(const cv::Mat& src);
      int getSDH(uint d, uint angle);
      int getSDH(const cv::Mat& src, uint d, uint angle);
      void computeFeatures(void);
      static int getSDH(const cv::Mat& src, SDH& sdh);
      static void computeFeatures(SDH& sdh);
      static void toCSV(std::vector<SDH>& sdh, std::string filename, unsigned int HEADER=0);
      static void toCSV_WriteHeader(std::string filename, unsigned int HEADER=0);
      void printFeatures(void);

    private:
      double
       *sumHist,
       *diffHist,
        Hist[2][511]
      ;
      static std::vector<SDH> _Lovdog_SDH_Features_Objects_;
  };
  typedef std::vector<SDH> SDHs;


};

#endif
