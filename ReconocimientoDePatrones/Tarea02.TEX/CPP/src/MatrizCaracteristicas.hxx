#ifndef __LOVDOG_MAT_CAR__
#define __LOVDOG_MAT_CAR__
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

typedef unsigned int uint;

namespace lovdog {
  typedef std::vector<cv::Point> Contour      ;
  typedef std::vector<Contour>   Contours     ;
  typedef std::vector<cv::Mat>   Gallery      ;

  typedef struct GLCMFeatures {
    GLCMFeatures();
    GLCMFeatures(const GLCMFeatures& f);
    friend std::ostream& operator<<(std::ostream& os, const GLCMFeatures& features);
    static constexpr int ANGLE[] = {0, 1, 2, 3, 4};
    static const int ANGLE_0 = 0;
    static const int ANGLE_45 = 1;
    static const int ANGLE_90 = 2;
    static const int ANGLE_135 = 3;
    static const int ANGLE_180 = 4;
    float mean;
    float variance;
    float IDM;
    float IDF;
    float entropy;
    float ASM;
    float energy;
    float correlation;
    float contrast;
    float homogeneity;
  } GLCMFeatures;

  typedef std::vector<GLCMFeatures> GLCMFeaturesVector;

  typedef struct GLRLMFeatures {
    GLRLMFeatures();
    GLRLMFeatures(const GLRLMFeatures& f);
    friend std::ostream& operator<<(std::ostream& os, const GLRLMFeatures& features);
    static constexpr int ANGLE[] = {0, 1, 2, 3};
    static const int ANGLE_0 = 0;
    static const int ANGLE_45 = 1;
    static const int ANGLE_90 = 2;
    static const int ANGLE_135 = 3;
    float SRE;     // Short Run Emphasis
    float LRE;     // Long Run Emphasis
    float GLN;     // Gray-Level Non-Uniformity
    float RLN;     // Run Length Non-Uniformity
    float RP;      // Run Percentage
    // Add more features as needed
  } GLRLMFeatures;

  class FeatureMatrix : cv::Mat {
  public:
    FeatureMatrix();
    FeatureMatrix(int rows, int cols, int type);
    FeatureMatrix(GLCMFeaturesVector& features);
    FeatureMatrix(const FeatureMatrix& m);
    FeatureMatrix(const cv::Mat& m);
    FeatureMatrix& operator=(const FeatureMatrix& m);
    FeatureMatrix& operator=(const cv::Mat& m);
    void add(GLCMFeatures& features);
    void rm(size_t index);
    std::string csvFormat() const;
  };

  void getMaskContours(cv::Mat& Mask, Contours& contours);
  void cropFromMaskContours(cv::Mat &src, Contours& contours, Gallery& dst, float radius=15, size_t randomCrops = 1);
  void getReferencePoint(Contour& points, cv::Rect& referenceRect);
  void calculateCropWindow(const cv::Rect& referencePoint, float radius, cv::Rect& cropRect, cv::RNG& rng);
  void showGallery(Gallery& gallery,std::string prefix="", std::string suffix="");
  void exportGallery(Gallery& gallery, std::string namePrefix="", std::string suffix="", std::string extension="png");
  /*GLCM Extraction*/
  void graycomatrix(cv::Mat& src, cv::Mat& glcm, uint distance, int angle);
  void getFeaturesFromGLCM(cv::Mat& glcm, GLCMFeatures& features);
  /*GLRL Extraction*/
  void computeGLRLM(const cv::Mat& src, cv::Mat& glrlm, int maxGrayLevel, int direction = 0);
  void extractGLRLMFeatures(const cv::Mat& glrlm, GLRLMFeatures& features);
};
#endif
