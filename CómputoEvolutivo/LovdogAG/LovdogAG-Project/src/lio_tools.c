#include "lio_tools.h"

// Para saber si a está en el array b
char en_array(void* elemento, void* array, size_t size, char tipo){
  size_t i=0;
  if(tipo=='i'){
      int* a=(int*)array;
      int* e=(int*)elemento;
      while(i<size) if(a[i++]==*e) return 1;
      return 0;
  }
  if(tipo=='I'){
    unsigned int* a=(unsigned int*)array;
    unsigned int* e=(unsigned int*)elemento;
    while(i<size) if(a[i++]==*e) return 1;
    return 0;
  }
  if(tipo=='s'){
      size_t* a=(size_t*)array;
      size_t* e=(size_t*)elemento;
      while(i<size) if(a[i++]==*e) return 1;
      return 0;
  }
  if(tipo=='c'){
    char* a=(char*)array;
    char* e=(char*)elemento;
    while(i<size) if(a[i++]==*e) return 1;
    return 0;
  }
  if(tipo=='C'){
    unsigned char* a=(unsigned char*)array;
    unsigned char* e=(unsigned char*)elemento;
    while(i<size) if(a[i++]==*e) return 1;
    return 0;
  }
  if(tipo=='f'){
    float* a=(float*)array;
    float* e=(float*)elemento;
    while(i<size) if(a[i++]==*e) return 1;
    return 0;
  }
  if(tipo=='d'){
    double* a=(double*)array;
    double* e=(double*)elemento;
    while(i<size) if(a[i++]==*e) return 1;
    return 0;
  }
  return 0;
}

void error_exit(char* message, int code){
  printf("%s\n", message);
  exit(code);
}

int error_return(char* message, int code){
#ifdef __LOVDOG_DEBUG_LOG__
  printf("%s\n", message);
#endif
  return code;
}
