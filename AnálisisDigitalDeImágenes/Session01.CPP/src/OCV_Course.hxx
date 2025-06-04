#ifndef OCVCOURSE_LOVDOG
#define OCVCOURSE_LOVDOG
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
int lovdog_Histograma2(const char*, const cv::Mat&);

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

#endif
