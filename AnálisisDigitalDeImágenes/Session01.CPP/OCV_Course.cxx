#include "OCV_Course.hxx"
#include <iostream>
#include <core.hpp>
#include <highgui.hpp>

int mainHolaMundo (int argc,char* argv[]){
  if(argc<2) { std::cout << "Requires an argument (image)"; return 1; }
  std::cout << "Checking OpenCV installation in PC /opencv/ \n";
  cv::Mat Input,InputClone,InputCopy;

  Input      = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);

  if(Input.empty()) { std::cout << "Image reading error"; return 1; }


  InputClone = Input.clone();
  Input.copyTo(InputCopy);


  imshow("Input image"                           , Input);
  imshow("Input image 2 (The Clone Attack)"      , InputClone);
  imshow("Input image 3 (The Rising of the Copy)", InputCopy);


  imwrite("FotoRobadaUwU.png",InputCopy);

  cv::waitKey(0);
  return 0;
}

int MatBasics(void){
  std::cout << "Basic Mat object arithmetics\n";
  std::cout << "Matrix Creation\n";
  std::cout << "Zeros Matrix (2,3)\n";
  cv::Mat matrix_zeros;
  matrix_zeros = cv::Mat::zeros(2,3,CV_32F);
  std::cout << matrix_zeros
            << std::endl << std::endl
  ;

  std::cout << "Ones Matrix (5,5)\n";
  cv::Mat matrix_ones;
  matrix_ones = cv::Mat::ones(5,5,CV_32F);
  std::cout << matrix_ones
            << std::endl << std::endl
  ;

  std::cout << "Identity Matrix (5,5)\n";
  cv::Mat matrix_identity;
  matrix_identity = cv::Mat::eye(5,5,CV_32F);
  std::cout << matrix_identity
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 32F\n";
  cv::Mat matrix_randomf = cv::Mat(2,4,CV_32F);
  randu(matrix_randomf,2,8);
  std::cout << matrix_randomf
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 8U\n";
  cv::Mat matrix_randomc = cv::Mat(2,4,CV_8U);
  randu(matrix_randomc,0,255);
  std::cout << matrix_randomc
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 8UC3\n";
  cv::Mat matrix_randomc3 = cv::Mat(2,4,CV_8UC3);
  randu(matrix_randomc3,0,255);
  std::cout << matrix_randomc3
            << std::endl << std::endl
  ;

  return 0;
}

int MatricOperations(){
  cv::Mat A = cv::Mat::eye ( cv::Size(3,2), CV_32F );
  cv::Mat B = cv::Mat::ones( cv::Size(3,2), CV_32F );
  cv::Mat C = cv::Mat::eye ( cv::Size(3,3), CV_32F );
  cv::Mat D = cv::Mat::ones( cv::Size(3,3), CV_32F );
  cv::Mat E = cv::Mat::eye ( cv::Size(2,3), CV_32F );
  std::cout<< "Given the matrices " << std::endl
           << "Matrix A"                << std::endl << A                 << std::endl << std::endl
           << "Matrix B"                << std::endl << B                 << std::endl << std::endl
           << "Matrix C"                << std::endl << C                 << std::endl << std::endl
           << "Matrix D"                << std::endl << D                 << std::endl << std::endl
           << "Matrix A+B"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix A+B"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix C+D"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix E+3"              << std::endl << E+3               << std::endl << std::endl
           << "Matrix E*2"              << std::endl << E*2               << std::endl << std::endl
           << "Matrix E*A"              << std::endl << E*A               << std::endl << std::endl
           << "Matrix (C+1).*(E+3)"     << std::endl << (E+1).mul(E+3)    << std::endl << std::endl
           << "Matrix A^T" << std::endl << A.t()                          << std::endl << std::endl
  ;
  return 0;
}

int ImaGenesis(){
  cv::Mat BlackCat, BlancChat, BlancChatSize, LaRayita, LosHunos;
  BlackCat      = cv::Mat::zeros(128,256,CV_8UC1);
  BlancChat     = cv::Mat::ones (128,256,CV_8UC1);
  BlancChatSize = cv::Mat::ones (cv::Size(128,256),CV_8UC1); 
  LaRayita      = cv::Mat::eye  (200,200,CV_8UC1);
  LosHunos      = cv::Mat::eye  (200,200,CV_8UC1);

  cv::imshow("BlackCat",BlackCat);
  cv::imshow("BlancChat",BlancChat*255);
  cv::imshow("BlancChat cv::Size",BlancChatSize*255);
  cv::imshow("La Rayita xd",LaRayita*255);
  cv::imshow("La Rayita Gris xd",(LosHunos+LaRayita)*(255/2));

  cv::waitKey();
  return 0;
}

int ImaGenesisDos(){
  std::cout << "El regreso de cv::Mat";

  cv::Mat ImagenNueva(cv::Size(128,256),CV_8UC1);
  // Incremento Gradual de intensidad

  //ImagenNueva.rows=20;

  for(int i=0; i < ImagenNueva.rows ; ++i)
    for(int j=0; j < ImagenNueva.cols; ++j)
      ImagenNueva.at<uchar>(i,j) = j;

  cv::imshow("Degradado A",ImagenNueva);

  cv::waitKey();
  return 0;
}
