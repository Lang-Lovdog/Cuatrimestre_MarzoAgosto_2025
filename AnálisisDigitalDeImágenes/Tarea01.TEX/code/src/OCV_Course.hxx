#ifndef OCVCOURSE_LOVDOG
#define OCVCOURSE_LOVDOG
#include <iostream>
#include <core.hpp>
#include <imgproc.hpp>
#include <opencv.hpp>
#include <highgui.hpp>

#define coutSEPARATOR  std::cout<<std::endl; for(int i=0;i<12;++i) std::cout << "-"; std::cout<<std::endl;

int mainHolaMundo (int argc,char* argv[]);
int MatBasics(void);
int MatricOperations();
int ImaGenesis();
int ImaGenesisDos();
int mainDatos(int argc, char** argv);
int ImagData(const cv::Mat&);
int videoStreaming(void);
int frameProcess(void);
int negateColour(const cv::Mat& , cv::Mat& );
int negateColour(cv::Mat&);
int frameROI(void);
int mainHistograma(int argc, char** argv);
int mainHistograma2(int argc, char** argv);
int lovdog_Histograma(const char*, const cv::Mat&);
int lovdog_HHistograma(const char*, const cv::Mat&);
int lovdogImagen2Histograma(const char* title, const cv::Mat& src, bool std_cum, size_t height, size_t width);
int lovdog_Histograma2(const char*, const cv::Mat&);

int mainNewHolaMundo (int argc,char* argv[]);
int lovdogBinarizacion(const cv::Mat& src, int threshValue, int threshCut, int threshType);
int lovdogStretchIt(const cv::Mat& src);
int lovdogEqualizeIt(const cv::Mat& src);

#define HISTOGRAM_SIZE 256
/* 
 * First argument is a Matrix which calculate the histogram from,
 * Second argument is the Matrix which will contain the histogram values
 * Third  argument, true for Probability Distribution, False for count
 * Fourth argument true for cumulative, false for standard
 */
int lovdogCalcHist(const cv::Mat& src, cv::Mat &histogram1D, bool cnt_pdf, bool std_cum);

/* Tool For Showing Histogram */
int lovdogShowHist(const cv::Mat& histogram, cv::Mat& ploutput, size_t width, size_t height);

// Filtros
int lovdogPasaAltas01(int argc,char** argv);
int CannyMain(int argc, char** argv);

// hsv
int lovdogHSV(int argc, char** argv);

#endif
