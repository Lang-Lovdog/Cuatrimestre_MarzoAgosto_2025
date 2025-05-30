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

#endif
