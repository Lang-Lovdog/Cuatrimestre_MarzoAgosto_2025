#ifndef __Lang_LIO_tools__
#define __Lang_LIO_tools__

#define TABULA for(int dkashdkjas=0; dkashdkjas<cout_tab; ++dkashdkjas) printf(" ");
#define TABULAR(CT) for(int dkashdkjas=0; dkashdkjas<CT; ++dkashdkjas) printf(" ");

#include <stdio.h>
#include <stdlib.h>

// Para saber si a está en el array b
char en_array(void* elemento, void* array, size_t size, char tipo);
/* Error exit print function */
void error_exit(char* message, int code);
/* Error exit print function */
int error_return(char* message, int code);

#endif
