#ifndef Lovdog_C_DICT_H
#define Lovdog_C_DICT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STR 2


typedef struct dict_T {
  char *key;
  char *value;
  struct dict_T *next;
} dict_T;

typedef struct dict {
  dict_T *head;
  size_t size;
} dict;
void  lovdog_dict_ch (dict* d,const char *key,char *value);
void  lovdog_dict_rm (dict* d,const char *key);
void  lovdog_dict_add(dict* d,char *key,char *value);
char* lovdog_dict_get(const dict* d,const char *key);
char  lovdog_dict_duplicate(const dict* d,const char *key);
void  lovdog_dict_wo_keshimasu(dict* d);
void  lovdog_dict_create_dict(dict* d);
unsigned int lovdog_dict_print_rec(dict *d);

#endif
