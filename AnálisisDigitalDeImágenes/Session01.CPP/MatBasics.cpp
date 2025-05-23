#include <iostream>
#include <core.hpp>

int MatBasics(void);

int main (void){
  MatBasics();
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
