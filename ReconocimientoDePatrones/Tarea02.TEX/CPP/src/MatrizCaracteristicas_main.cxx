#include "MatrizCaracteristicas_main.hxx"
#include "MatrizCaracteristicas.hxx"

int MAIN(int argc, char const *argv) {
  // Import image in GrayScale
  cv::Mat src, glcm[3*4], glrlm[3*4];
  lovdog::GLCMFeatures  features0[3*4];
  lovdog::GLRLMFeatures features1[3*4];
  uint distance[3] = { 1,  3,  7 };
  int i,j;

  src = cv::imread(argv, CV_8UC1);
  if(src.empty()){
    std::cerr << "Usage: " << argv[0] << " <image>" << std::endl;
    return -1;
  }

  i = j = 0;
  while(j<4){
    while(i<3){
      // GLCM Extraction
      lovdog::graycomatrix(src, glcm[i*3+j], distance[i], lovdog::GLCMFeatures::ANGLE[j]);
      lovdog::getFeaturesFromGLCM(glcm[i*3+j], features0[i*3+j]);
      ++i;
    }
    ++j; i=0;
  }
  j = 0;
  while(j<3){
    lovdog::computeGLRLM(src, glrlm[i*3+j], 256, 0);
    lovdog::extractGLRLMFeatures(glrlm[i*3+j], features1[i*3+j]);
    ++j; i=0;
  }
  return 0;
}
