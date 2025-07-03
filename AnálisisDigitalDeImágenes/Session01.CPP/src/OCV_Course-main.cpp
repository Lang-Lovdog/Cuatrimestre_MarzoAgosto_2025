#include <iostream>
#include <highgui.hpp>
#include <core.hpp>
#include "OCV_Course.hxx"

int main (int argc, char* argv[]){
  //std::cout << "Read Image"<< std::endl; mainHolaMundo(argc,argv);
  //std::cout << "MatrixOperations"<< std::endl; MatricOperations();
  //std::cout << "ImaGenesis" << std::endl; ImaGenesis();
  //std::cout << "ImaGenesis2" << std::endl; ImaGenesisDos();
  //std::cout << "Image Data " << std::endl; mainDatos(argc, argv);
  //std::cout << "Camera capturing\n\nI can see you"; videoStreaming();
  //std::cout << "Camera capturing\n\nI can see you by 2"; frameProcess();
  //std::cout << "Camera capturing\n\nI can see you by 2 and kinda one"; frameROI();
  //std::cout << "La vie en gris" << std::endl; mainHistograma(argc,argv);
  //std::cout << "La vie en gris 2 el histograma, regresa" << std::endl; mainHistograma2(argc,argv);
  //std::cout << "Binarización :O" << std::endl; mainNewHolaMundo(argc,argv);
  //std::cout << "Pasa altas >:C" <<std::endl; lovdogPasaAltas01(argc,argv);
  //std::cout << "Canny" <<std::endl; CannyMain(argc,argv);
  std::cout<< "HSV DOMAIN :0"<<std::endl; lovdogHSV(argc, argv);
  return 0;
}
