#include <iostream>
#include <core.hpp>
#include <highgui.hpp>

int main (int argc,char* argv[]){
  if(argc<2) { std::cout << "Requires an argument (image)"; return 1; }
  std::cout << "Checking OpenCV installation in PC /opencv/ \n";
  cv::Mat Input;
  Input = cv::imread(argv[1]);
  if(Input.empty()) { std::cout << "Image reading error"; return 1; }
  imshow("Input image", Input);
  cv::waitKey(0);
  return 0;
}
