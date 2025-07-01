#include "dibujo.hxx"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// Este programa está diseñado para encontrar los bordes de una imágen y hacer una limpieza de dibujo
// El código es algo sencillo y únicamente se sostendrá en la forma de detección de bordes para una imágen con 
// la finalidad de encontrar las líneas delimitantes más importantes de la imágen haciendo posible su paso a dibujo.

int main (int argc,char **argv){
  if(argc < 2) { std::cerr << "Usage " << argv[0] << "image.ext"; return 1; }
  cv::Mat lImage = cv::imread(argv[1],CV_8UC1);
  if(lImage.empty()){std::cerr << "Error opening the image" << std::endl; return 1;}

  cv::Mat EdgeImage, EdgeEqualized, Edge, kernel;

  GetEdge(lImage,EdgeImage);
  lovdogEqualizeIt(lImage, lImage);
  GetEdge(lImage,EdgeEqualized, false);
  showIt(EdgeImage,"salida/"+std::string(argv[1])+"_Edge",false);
  showIt(EdgeEqualized,"salida/"+std::string(argv[1])+"_Equalized - Edge",true);
  //cv::waitKey(0);
}
