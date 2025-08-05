#include "MatrizCaracteristicas_main.hxx"
#include <iostream>

int main(int argc, char const **argv) {
  if(argc < 2){
    std::cerr << "Usage: " << argv[0] << " <image>" << std::endl;
    return -1;
  }
  size_t i;

  i=0;

  while(i<(size_t)argc){
    MAIN(argc, *(argv+i));
    ++i;
  }

  return 0;
}

