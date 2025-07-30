#include "ColorTransfer.hxx"

int main(int argc, char** argv){
  if(argc<3){
    std::cout << "Requires at least two arguments: image  reference  [output]" << std::endl; 
    return 1;
  }
  lovdog::mainColorTransfer(argc, argv);
  return 0;
}
