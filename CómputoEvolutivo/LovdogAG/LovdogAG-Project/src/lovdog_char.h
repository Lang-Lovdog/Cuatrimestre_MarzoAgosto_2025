#ifndef __LOVDOG_CHAR__
#define __LOVDOG_CHAR__

#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include "lio_tools.h"

struct lovdog_char{
  char c;
  struct lovdog_char* ant;
  struct lovdog_char* sig;
  size_t indice;
};

struct lovdog_texto{
  struct lovdog_char* primer_letra;
  struct lovdog_char* ultima_letra;
  struct lovdog_char* actual;
  struct lovdog_char** tokens;
  char *token_delimitador;
  size_t* token_longitud;
  size_t token_actual;
  size_t token_cantidad;
  size_t longitud;
};

typedef struct lovdog_char lovdog_char;
typedef struct lovdog_texto lovdog_texto;

// Getline para lovdog_char
int lovdog_getline(FILE *flujo, struct lovdog_texto** texto);
// Getline para lovdog_char delimited by...
int lovdog_getlined(FILE *flujo, struct lovdog_texto** texto, const char* delim);
// Getlinec para puntero char
int lovdog_getlinec(FILE *flujo, char** texto);
// Getlinec para puntero char delimited by...
int lovdog_getlinedc(FILE *flujo, char** texto, const char* delim);
// Escribe el contenido de lovdog texto en un archivo
char lovdog_texto_fwrite(FILE *flujo, struct lovdog_texto** texto);
// Escribe el contenido de un archivo en lovdog texto
char lovdog_texto_fread(FILE *flujo, struct lovdog_texto** texto);
// Convert lovdog_texto a char*
char* lovdog_texto_str(struct lovdog_texto** texto);
// Convert token a char*
char* lovdog_textok_str(struct lovdog_texto** texto, size_t token_index);
// Convert lovdog_texto from index a to index b to char*
char* lovdog_texto_str_rango(struct lovdog_texto** texto, size_t i1, size_t i2);
// Lovdog texto a entero/float/double
void* lovdog_texto_int(struct lovdog_texto** texto, char tipo);
// Lovdog token a entero/float/double
void* lovdog_textok_int(struct lovdog_texto** texto, size_t k, char tipo);
// Concatenar cadenas
char lovdog_texto_strappend(struct lovdog_texto** texto, char* str);
// Concatenar lovdog_texto
char lovdog_texto_textoappend(struct lovdog_texto** t, struct lovdog_texto* str);
// Insertar cadena en la posición del índice
char lovdog_texto_strinsert(struct lovdog_texto** texto, char* str, size_t posicion);
// Concatenar cadena como token
char lovdog_textok_append(struct lovdog_texto** texto, char* str);
// Concatenar cadena y carácter
char lovdog_texto_charappend(struct lovdog_texto** texto, char c);
// Añadir carácter en la posición del índice
char lovdog_texto_charinsert(struct lovdog_texto** texto, char c, size_t posicion);
// Circular lovdog_texto
char lovdog_texto_circula(struct lovdog_texto** texto, char reversa);
/* Mueve el iterador al carácter en la posición del índice */
char lovdog_char_en_texto(struct lovdog_texto** texto, size_t posicion);
// Creación e inicialización de lovdog_char
struct lovdog_char* lovdog_charinit();
// Liberación de lovdog_char
char lovdog_charfree(struct lovdog_char** c);
//Creación e inicialización de lovdog_texto
struct lovdog_texto* lovdog_textoinit();
// Liberación de lovdog_texto
char lovdog_textofree(struct lovdog_texto** t);
// Imprimir lovdog_texto
void lovdog_texto_cout(struct lovdog_texto* t);
// Imprime tokens de una estructura lovdog_texto
// si token=-1, imprime todos
char lovdog_textok_cout(struct lovdog_texto* t, long int token);
// Elimina carácteres repetidos al inicio y al final del texto
char lovdog_texto_trim(struct lovdog_texto** texto, char caracter);
// Elimina carácteres repetidos al inicio y al final del texto
char lovdog_textok_trim(struct lovdog_texto** texto, char caracter);
/* Elimina el carácter en la posición del índice*/
char lovdog_texto_rm(struct lovdog_texto** texto, size_t modo);
/* Elimina el carácter en la posición del índice si cumple la condición.
 * Recibe 3 argumentos, el texto, la posición del carácter y el carácter
 * al que debe ser igual para ser eliminado */
char lovdog_texto_rm_si(struct lovdog_texto** texto, size_t pos, char condicion);
/* Busca char en lovdog_texto */
char lovdog_texto_busca(struct lovdog_texto** texto, char c, char reversa);
/* Divide en tokens */
char lovdog_texto_token(struct lovdog_texto** texto, char* delimitador);
/* Comparar lovdog_texto */
char lovdog_texto_cmp(struct lovdog_texto* t1, struct lovdog_texto* t2);
/* Comparar lovdog_texto tokens */
char lovdog_textok_cmp(struct lovdog_texto* t1, size_t i1, struct lovdog_texto* t2, size_t i2);
/* Comparar lovdog_texto  con tokens */
char lovdog_texto_token_cmp(struct lovdog_texto* t1, struct lovdog_texto* t2, size_t i2);
/* Comparar lovdog_char */
char lovdog_char_cmp(struct lovdog_char* c1, struct lovdog_char* c2);
/* Copiar lovdog_char, c1 es el apuntador de destino;
 * si, puntero=-1, genera una nueva entidad;
 * si, puntero=0, comparte punteros;
 * si puntero=1, c1 se queda con los punteros y c2 se queda en null */
char lovdog_texto_cpy(struct lovdog_texto** c1, struct lovdog_texto** c2, char puntero);
/* Restablece tokens */
char lovdog_textok_reset(struct lovdog_texto** texto);
/* Compara todos los tokens para verificar si el texto a comparar ya se encuentra en la estructura.
 * El primer elemento es el texto a comparar y el segundo elemento es el texto tokenizado.
 * Esta función retorna la cantidad de tokens iguales a t1.*/
long int lovdog_textok_existe(struct lovdog_texto* t1, struct lovdog_texto* t2);
/* Compara todos los tokens para verificar si el texto a comparar ya se encuentra en la estructura.
 * El primer elemento es el texto a comparar y el segundo elemento es el texto tokenizado.
 * Esta función posiciona el iterador de t1 en el primer token igual a t2 
 * y retorna la posición correspondiente*/
long int lovdog_textok_busca(struct lovdog_texto* t1, struct lovdog_texto* t2);
/* Manejo de prefijos y sufijos, aquí, se manejarán los sufijos y prefijos en la 
 * estructura de lovdog_texto. Para lograrlo, se utilizará una cadena char que
 * indique el prefijo y otra char que indique el sufijo. */
/* Verificar si el sufijo y el prefijo se encuentran en el texto 
 * si el prefijo es NULL, solo se verificará el sufijo, del mismo modo, 
 * si el sufijo es NULL, solo se verificará el prefijo*/
char lovdog_texto_prefijo_sufijo_si(struct lovdog_texto** texto, const char* prefijo, const char* sufijo);
/* Verificar si el sufijo y el prefijo se encuentran en el token 
 * si el prefijo es NULL, solo se verificará el sufijo, del mismo modo, 
 * si el sufijo es NULL, solo se verificará el prefijo*/
char lovdog_textok_prefijo_sufijo_si(struct lovdog_texto** texto, size_t token, const char* prefijo, const char* sufijo);
/* Regresa una cadena sin el prefijo y sin el sufijo especificados */
char* lovdog_texto_prefijo_sufijo_sin(struct lovdog_texto** texto, const char* prefijo, const char* sufijo);
/* Regresa una cadena sin el prefijo y sin el sufijo especificados versión token */
char* lovdog_textok_prefijo_sufijo_sin(struct lovdog_texto** texto, size_t token, const char* prefijo, const char* sufijo);
/* Agrega un prefijo y un sufijo a lovdog_texto */
char lovdog_texto_prefijo_sufijo_add(struct lovdog_texto** texto, const char* prefijo, const char* sufijo);
/* Agrega un prefijo y un sufijo a lovdog_texto versión token */
char lovdog_textok_prefijo_sufijo_add(struct lovdog_texto** texto, size_t token, const char* prefijo, const char* sufijo);
/* Lovdog textok a texto */
lovdog_texto* lovdog_textok_texto(struct lovdog_texto** texto, size_t token);

#endif
