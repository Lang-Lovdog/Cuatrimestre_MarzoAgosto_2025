#include "lovdog_char.h"

// Getline para lovdog_char
int lovdog_getline(FILE *f, struct lovdog_texto** t){
  if(!f) return -1;
  char c;
  char fin=1;
  while(fscanf(f,"%c",&c)!=EOF){
    if(c=='\n') fin=0;
    if(!fin) break;
    lovdog_texto_charappend(t,c);
  }
  return fin;
}

// Getline para lovdog_char delimited by...
int lovdog_getlined(FILE *f, struct lovdog_texto** t, const char* d){
  if(!f) return -1;
  if(!d) return -1;
  char c,fin=1;
  while(fscanf(f,"%c",&c)!=EOF){
    for(size_t i=0; d[i]!='\0'; ++i) if (c==d[i]) fin=0;
    if(fin==0) break;
    lovdog_texto_charappend(t,c);
  }
  return fin;
}

// Getlinec para puntero char
int lovdog_getlinec(FILE *flujo, char** t){
  if(!flujo) return -1;
  if(!*t) return -1;
  lovdog_texto* lt = lovdog_textoinit();
  if(!lt) return -1;
  lovdog_getline(flujo, &lt);
  *t = lovdog_texto_str(&lt);
  return 0;
}

// Getlinec para puntero char delimited by...
int lovdog_getlinedc(FILE *flujo, char** texto, const char* delim){
  if(!flujo) return -1;
  if(!delim) return -1;
  lovdog_texto* lt = NULL;
  int res=0;
  res=lovdog_getlined(flujo, &lt, delim);
  if(res) return res;
  *texto = lovdog_texto_str(&lt);
  lovdog_textofree(&lt);
  return 0;
}

// Escribe el contenido de lovdog texto en un archivo
char lovdog_texto_fwrite(FILE *flujo, struct lovdog_texto** texto){
  if(!flujo) return -1;
  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -1;
  if(!(*texto)->ultima_letra) return -1;
  if(!(*texto)->longitud) return -1;
  struct lovdog_char* lc = (*texto)->primer_letra;
  while(lc){
    fprintf(flujo, "%c", lc->c);
    lc = lc->sig;
  }
  return 0;
}

// Escribe el contenido de un archivo en lovdog texto
char lovdog_texto_fread(FILE *f, struct lovdog_texto** t){
  if(!f) return -1;
  char c;
  char fin=0;
  while(fscanf(f,"%c",&c)!=EOF)
    fin=lovdog_texto_charappend(t,c);
  return fin;
}

// Convert lovdog_texto a char*
char* lovdog_texto_str(struct lovdog_texto** t){
  if(!*t) return NULL;
  if(!(*t)->primer_letra) return NULL;
  if(!(*t)->ultima_letra) return NULL;
  if(!(*t)->longitud) return NULL;
  char* str = (char*)malloc(sizeof(char)*(*t)->longitud+1);
  struct lovdog_char* lc = (*t)->primer_letra;
  for(size_t i=0; i<(*t)->longitud; ++i){
    str[i] = lc->c;
    lc = lc->sig;
  }
  str[(*t)->longitud] = '\0';
  return str;
}

// Convert lovdog_texto from index a to index b to char*
char* lovdog_texto_str_rango(struct lovdog_texto** texto, size_t i1, size_t i2){
  if(!*texto) return NULL;
  if(!(*texto)->primer_letra) return NULL;
  if(!(*texto)->ultima_letra) return NULL;
  if(!(*texto)->longitud) return NULL;
  lovdog_char *actual=(*texto)->actual;
  lovdog_char_en_texto(texto, i1);
  lovdog_char* lc = (*texto)->actual;
  lovdog_char_en_texto(texto, i2);
  lovdog_char* lk = (*texto)->actual;
  (*texto)->actual = actual;
  size_t longitud = lk->indice-lc->indice+2;
  char* salida = (char*) malloc(sizeof(char)*longitud);
  size_t i=0;
  while(lc!=lk->sig){
    salida[i] = lc->c;
    lc = lc->sig; ++i;
  }
  salida[longitud-1] = '\0';
  return salida;
}

// Convert lovdog_texto token a char*
char* lovdog_textok_str(struct lovdog_texto** t, size_t i){
  if(!*t) return NULL;
  if(!(*t)->primer_letra) return NULL;
  if(!(*t)->ultima_letra) return NULL;
  if(!(*t)->longitud) return NULL;
  if(!(*t)->tokens) return NULL;
  if(!(*t)->token_longitud) return NULL;
  if(i>=(*t)->token_cantidad) return NULL;
  size_t longitud = (*t)->token_longitud[i];
  char* str = (char*) malloc(sizeof(char)*longitud+1);
  struct lovdog_char* lc = (*t)->tokens[i];
  for(size_t n=0; n<longitud; ++n){
    str[n] = lc->c;
    lc = lc->sig;
  }
  str[longitud] = '\0';
  return str;
}

// Lovdog texto a entero/float/double
void* lovdog_texto_int(struct lovdog_texto** t, char i){
  if(!*t) return NULL;
  if(!(*t)->primer_letra) return NULL;
  if(!(*t)->ultima_letra) return NULL;
  if(!(*t)->longitud) return NULL;
  // Si hay otros elementos que no sean digitos, retorna NULL
  struct lovdog_char *lc=NULL; size_t n=0;
  char digitos[12]="1234567890.", hay_punto=0, *final;

  lc=(*t)->primer_letra;
  while(n<(*t)->longitud){
    if(!en_array(&lc->c, digitos, strlen(digitos), 'c')){
      return NULL;
    }
    if(lc->c=='.') hay_punto++;
    if(hay_punto>1){
      return NULL;
    }
    lc=lc->sig; ++n;
  }
  if(i=='i'){
    int *salida=(int*)malloc(sizeof(int));
    final=lovdog_texto_str(t);
    *salida=atoi(final);
    free(final);
    return salida;
  }
  if(i=='f'){
    float *salida=(float*)malloc(sizeof(float));
    final=lovdog_texto_str(t);
    *salida=atof(final);
    free(final);
    return salida;
  }
  if(i=='d'){
    double *salida=(double*)malloc(sizeof(double));
    final=lovdog_texto_str(t);
    *salida=atof(final);
    free(final);
    return salida;
  }
  if(i=='s'){
    size_t *salida=(size_t*)malloc(sizeof(size_t));
    final=lovdog_texto_str(t);
    sscanf(final, "%zu", salida);
    free(final);
    return salida;
  }
  return NULL;
}

// Lovdog token a entero/float/double
void* lovdog_textok_int(struct lovdog_texto** t, size_t k, char i){
  if(!*t) return NULL;
  if(!(*t)->primer_letra) return NULL;
  if(!(*t)->ultima_letra) return NULL;
  if(!(*t)->longitud) return NULL;
  if(!(*t)->tokens) return NULL;
  if(!(*t)->token_longitud) return NULL;
  if(!(*t)->token_cantidad) return NULL;
  // Si hay otros elementos que no sean digitos, retorna NULL
  struct lovdog_char *lc=NULL; size_t n=0;
  char digitos[12]="1234567890.", hay_punto=0, *final;

  lc=(*t)->tokens[k];
  while(n<(*t)->token_longitud[k]){
    if(!en_array(&lc->c, digitos, strlen(digitos), 'c')){
      return NULL;
    }
    if(lc->c=='.') hay_punto++;
    if(hay_punto>1){
      return NULL;
    }
    lc=lc->sig; ++n;
  }
  if(i=='i'){
    int *salida=(int*)malloc(sizeof(int));
    final=lovdog_textok_str(t, k);
    *salida=atoi(final);
    free(final);
    return salida;
  }
  if(i=='f'){
    float *salida=(float*)malloc(sizeof(float));
    final=lovdog_textok_str(t, k);
    *salida=atof(final);
    free(final);
    return salida;
  }
  if(i=='d'){
    double *salida=(double*)malloc(sizeof(double));
    final=lovdog_textok_str(t, k);
    *salida=atof(final);
    free(final);
    return salida;
  }
  if(i=='s'){
    size_t *salida=(size_t*)malloc(sizeof(size_t));
    final=lovdog_texto_str(t);
    sscanf(final, "%zu", salida);
    free(final);
    return salida;
  }
  return NULL;
}

// Concatenar cadenas
char lovdog_texto_strappend(struct lovdog_texto** t, char* str){
  size_t str_longitud=strlen(str);
  for(size_t i=0; i<str_longitud; ++i) lovdog_texto_charappend(t, str[i]);
  return 0;
}

// Concatenar lovdog_texto
char lovdog_texto_textoappend(struct lovdog_texto** t, struct lovdog_texto* str){
  if(!str) return -1;
  size_t str_longitud=str->longitud;
  struct lovdog_char *lt = str->primer_letra;
  for(size_t i=0; i<str_longitud; ++i){
    lovdog_texto_charappend(t, lt->c);
    lt=lt->sig;
  }
  return 0;
}

// Insertar cadena en la posición del índice
char lovdog_texto_strinsert(struct lovdog_texto** texto, char* str, size_t p){
  for(size_t i=0; i<strlen(str); ++i) lovdog_texto_charinsert(texto, str[i], p+i);
  return 0;
}

// Concatenar cadena como token
char lovdog_textok_append(struct lovdog_texto** t, char* str){
  if(!*t) return -1;
  if(!(*t)->tokens) return -1;
  if(!(*t)->token_longitud) return -1;
  if(!(*t)->token_cantidad) return -1;
  if(!(*t)->token_delimitador) return -1;
  if(!str) return -2;
  if(!strlen(str)) return -2;
  lovdog_char *actual=(*t)->actual;
  size_t str_longitud=strlen(str);
  lovdog_char **tokens=(*t)->tokens;
  size_t *token_longitud=(*t)->token_longitud;
  (*t)->token_longitud=NULL; (*t)->tokens=NULL;
  (*t)->tokens=(lovdog_char**)malloc(sizeof(lovdog_char*)*((*t)->token_cantidad+1));
  (*t)->token_longitud=(size_t*)malloc(sizeof(size_t)*((*t)->token_cantidad+1));
  for(size_t i=0; i<(*t)->token_cantidad; ++i){
    (*t)->token_longitud[i]=token_longitud[i];
    (*t)->tokens[i]=tokens[i];
  }(*t)->token_cantidad++;
  free(token_longitud); free(tokens);

  lovdog_texto_charappend(t, *(*t)->token_delimitador);
  for(size_t i=0; i<str_longitud; ++i) lovdog_texto_charappend(t, str[i]);
  lovdog_texto_circula(t, -1);
  lovdog_texto_busca(t, *(*t)->token_delimitador, 1);
  lovdog_texto_circula(t, 0);
  (*t)->tokens[(*t)->token_cantidad-1]=(*t)->actual;
  (*t)->token_longitud[(*t)->token_cantidad-1]=str_longitud;
  
  (*t)->actual=actual;
  return 0;
}

// Concatenar cadena y carácter
char lovdog_texto_charappend(struct lovdog_texto** t, char c){
  struct lovdog_char* lc = lovdog_charinit();
  if(!lc) return -1;
  lc->c = c;
  if(!*t){
    (*t) = lovdog_textoinit();
    if(!(*t)) error_return("Error al crear estructura de texto", -1);
    lc->ant = NULL;
    (*t)->primer_letra = lc;
    goto lovdog_texto_charappend_lastwords;
  }
  struct lovdog_char* iterador=(*t)->primer_letra;
  if(!iterador){
    lc->ant = NULL;
    (*t)->primer_letra = lc;
    goto lovdog_texto_charappend_lastwords;
  }
  lc->ant = (*t)->ultima_letra;
  (*t)->ultima_letra->sig = lc;
lovdog_texto_charappend_lastwords: 
  lc->sig = NULL;
  lc->indice = (*t)->longitud;
  (*t)->ultima_letra = lc;
  ++(*t)->longitud;
  return 0;
}

// Añadir carácter en la posición del índice
char lovdog_texto_charinsert(struct lovdog_texto** t, char c, size_t p){
  struct lovdog_char* lc = lovdog_charinit();
  if(!lc) return -1;
  lc->c = c;
  if(!*t){
    (*t) = lovdog_textoinit();
    if(!(*t)) error_return("Error al crear estructura de texto", -1);
    lc->sig = lc->ant = NULL;
    (*t)->ultima_letra = (*t)->primer_letra = lc;
    lc->indice = (*t)->longitud = 0;
    ++(*t)->longitud;
    return 0;
  }
  struct lovdog_char* iterador=(*t)->primer_letra;
  if(!iterador){
    lc->sig = lc->ant = NULL;
    (*t)->ultima_letra = (*t)->primer_letra = lc;
    lc->indice = (*t)->longitud = 0;
    ++(*t)->longitud;
    return 0;
  }
  // Inserción en la posición
  /// Posicionamiento del iterador
  struct lovdog_char* actual=(*t)->actual;
  if(!lovdog_char_en_texto(t, p)) return -2;
  iterador=(*t)->actual;
  (*t)->actual=actual;
  /// Inserción del nodo lovdog_char
  /// Conectamos el nodo nuevo a los nodos correspondientes
  if(!iterador->ant) (*t)->primer_letra = lc;
  lc->ant=iterador->ant;
  lc->sig = iterador;
  /// Reconectamos los nodos al nuevo
  if(!iterador->ant) iterador->ant->sig = lc;
  iterador->ant = lc;
  /// Reacomodamos índices
  lc->indice=iterador->indice;
  while(iterador){
    iterador->indice++;
    iterador=iterador->sig;
  }
  return 0;
}

// Circular lovdog_texto, reversa=-1 para reiniciar el iterador
char lovdog_texto_circula(struct lovdog_texto** texto, char reversa){
  if(!*texto) return -1;
  if(reversa==-1){ (*texto)->actual=NULL; return 0; }
  if(!reversa){
    if((*texto)->actual) if(!(*texto)->actual->sig) return 1;
    (*texto)->actual=(*texto)->actual? (*texto)->actual->sig : (*texto)->primer_letra;
    return 0;
  }
  if(reversa){
    if((*texto)->actual) if(!(*texto)->actual->ant) return 1;
    (*texto)->actual=(*texto)->actual? (*texto)->actual->ant : (*texto)->ultima_letra;
    return 0;
  }
  return 0;
}

// Mueve el iterador al carácter en la posición del índice
char lovdog_char_en_texto(struct lovdog_texto** texto, size_t posicion){
  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -1;
  if(!(*texto)->ultima_letra) return -1;
  if(posicion>=(*texto)->longitud) return -2;
  (*texto)->actual = (*texto)->primer_letra;
  while(1){
    if((*texto)->actual->indice>posicion){
      lovdog_texto_circula(texto, 1);
      continue;
    }
    if((*texto)->actual->indice<posicion){
      lovdog_texto_circula(texto, 0);
      continue;
    }
    break;
  }
  return 0;
}

// Creación e inicialización de lovdog_char
struct lovdog_char* lovdog_charinit(){
  struct lovdog_char* lc =
    (lovdog_char*)malloc(sizeof(struct lovdog_char));
  lc->c = '\0';
  lc->ant = NULL;
  lc->sig = NULL;
  return lc;
}

// Liberación de lovdog_char
char lovdog_charfree(struct lovdog_char** c){
  if(!*c) return 0;
  if((*c)->ant)
    error_return("Nodos enlazados no vacíos (anterior)", -1);
  if((*c)->sig)
    error_return("Nodos enlazados no vacíos (siguiente)", -1);
  free(*c);
  *c = NULL;
  return 0;
}

//Creación e inicialización de lovdog_texto
struct lovdog_texto* lovdog_textoinit(){
  struct lovdog_texto* lt = (lovdog_texto*) malloc(sizeof(struct lovdog_texto));
  lt->primer_letra = NULL;
  lt->ultima_letra = NULL;
  lt->actual = NULL;
  lt->tokens = NULL;
  lt->token_longitud = NULL;
  lt->token_delimitador = NULL;
  lt->longitud = 0;
  lt->token_cantidad = 0;
  lt->token_actual = 0;
  return lt;
}

// Liberación de lovdog_texto
char lovdog_textofree(struct lovdog_texto** t){
  if(!*t) return -1;
  struct lovdog_char* lc, *lk;
  size_t i;
  lc = (*t)->primer_letra;
  lk = lc->sig;
  if((*t)->tokens){
    i=0;
    while(i<(*t)->token_cantidad){
      (*t)->tokens[i]=NULL; ++i;
    }
  }
  if((*t)->tokens) free((*t)->tokens);
  if((*t)->token_longitud) free((*t)->token_longitud);
  if((*t)->token_delimitador) free((*t)->token_delimitador);
  if((*t)->token_cantidad) (*t)->token_cantidad=0;
  while(1){
    lc->ant=lc->sig=NULL;
    if(lovdog_charfree(&lc))
      return error_return("No se pudo liberar el puntero", -1);
    lc = lk;
    if(!lc) break;
    lk = lc->sig;
  }
  free(*t);
  *t = NULL;
  return 0;
}

// Imprime una estructura lovdog_texto
void lovdog_texto_cout(struct lovdog_texto* t){
  if(!t) return;
  struct lovdog_char* lc = t->primer_letra;
  while(lc){
    printf("%c", lc->c);
    lc = lc->sig;
  }
}

// Imprime tokens de una estructura lovdog_texto
// si token=-1, imprime todos
char lovdog_textok_cout(struct lovdog_texto* t, long int token){
  if(!t) return -1;
  if(token>=(long int)t->token_cantidad) return -2;
  if(!t->tokens) return -3;
  if(!t->token_longitud) return -3;
  if(!t->token_cantidad) return -3;
  size_t i=0, ch=0, limite=0;
  struct lovdog_char* lc = NULL;
  if(token>-1){
    lc = t->tokens[token];
    ch=0; limite=t->token_longitud[token];
    while(ch<limite){
      printf("%c", lc->c);
      lc = lc->sig; ++ch;
    } return 0;
  }
  size_t cout_tab=(-2-token)>0?(-2-token):0;
  // Imprimir todos los tokens
  TABULA;
  printf("{\n");
  i=0;
  while(i<t->token_cantidad){
    TABULA;
    printf(" %4llu:  ",i);
    TABULA;
    lovdog_textok_cout(t,i++);
    printf("\n");
  }
    TABULA;
  printf("}");
  return 0;
}

// Quita caracteres redundantes que rodean al texto
char lovdog_texto_trim(struct lovdog_texto** t, char c){
  if(!*t) return -1;
  if(!(*t)->primer_letra) return -1;
  if(!(*t)->ultima_letra) return -1;
  // Iniciamos búsqueda
  while(!lovdog_texto_rm_si(t,0,c));
  // Siguiente busqueda
  while(!lovdog_texto_rm_si(t,(*t)->longitud-1,c));
  return 0;
}

// Quita caracteres redundantes que rodean al texto del token
char lovdog_textok_trim(struct lovdog_texto** t, char c){
  if(!*t) return -1;
  if(!(*t)->primer_letra) return -1;
  if(!(*t)->ultima_letra) return -1;
  if(!(*t)->tokens) return -2;
  if(!(*t)->token_longitud) return -2;
  if(!(*t)->token_cantidad) return -2;
  struct lovdog_char* lc=NULL;
  size_t 
    i=0,
    token=0,
    limiti=0,
    limitok=(*t)->token_cantidad;
  // Iniciamos búsqueda (ascendente)
  while(token<limitok){
    lc=(*t)->tokens[token]; i=0;
    limiti=(*t)->token_longitud[token];
    while(i<limiti){
      if(lc->c!=c) break;
      lc=lc->sig;
      --(*t)->token_longitud[token];
      ++i;
    }
    (*t)->tokens[token]=lc;
    ++token;
  }
  // Iniciamos búsqueda (ascendente)
  token=0; i=0; limitok=0; limiti=0;
  while(token<limitok){
    lc=(*t)->tokens[token]; i=0;
    limiti=(*t)->token_longitud[token];
    // Avanzamos hasta el último elemento del token
    while(i<limiti){ lc=lc->sig; ++i; }
    // Iteramos en reversa hasta encontrar un carácter distinto al delimitador
    i=0;
    while(i<limiti){
      if(lc->c!=c) break;
      lc=lc->ant;
      --(*t)->token_longitud[token];
      ++i;
    }
    ++token;
  }
  return 0;
}

/* Elimina el carácter en la posición del índice*/
char lovdog_texto_rm(struct lovdog_texto** texto, size_t modo){
  struct lovdog_char
    *actual=NULL,
    *objetivo=NULL,
    *anterior=NULL,
    *siguient=NULL;
  char
    estado_circular=0;

  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -1;
  if(!(*texto)->ultima_letra) return -1;

  actual= (*texto)->actual;
  estado_circular=lovdog_char_en_texto(texto, modo);
  if(estado_circular) return estado_circular;

  objetivo = (*texto)->actual;
  anterior = objetivo->ant;
  siguient = objetivo->sig;

  if(!anterior) (*texto)->primer_letra = siguient;
  if(!siguient) (*texto)->ultima_letra = anterior;

  if(anterior) anterior->sig = siguient;
  if(siguient) siguient->ant = anterior;
  if(objetivo==actual) (*texto)->actual = anterior;
  else (*texto)->actual = actual;

  objetivo->ant = NULL;
  objetivo->sig = NULL;
  if(lovdog_charfree(&objetivo))
    return error_return("No se pudo liberar el puntero", -1);
  // Reacomodar indices de la tabla
  while (siguient) {
    siguient->indice--;
    siguient = siguient->sig;
  }
  --(*texto)->longitud;
  return 0;
}

/* Elimina el carácter en la posición del índice si cumple la condición*/
char lovdog_texto_rm_si(struct lovdog_texto** texto, size_t pos, char c){
  struct lovdog_char
    *actual=NULL,
    *objetivo=NULL,
    *anterior=NULL,
    *siguient=NULL;
  char
    estado_circular=0;

  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -1;
  if(!(*texto)->ultima_letra) return -1;

  actual= (*texto)->actual;
  estado_circular=lovdog_char_en_texto(texto, pos);
  if(estado_circular) return estado_circular;

  objetivo = (*texto)->actual;
  if(objetivo->c != c){
    (*texto)->actual = actual;
    return 1;
  }
  anterior = objetivo->ant;
  siguient = objetivo->sig;

  if(!anterior) (*texto)->primer_letra = siguient;
  if(!siguient) (*texto)->ultima_letra = anterior;

  if(anterior) anterior->sig = siguient;
  if(siguient) siguient->ant = anterior;
  if(objetivo==actual) (*texto)->actual = anterior;
  else (*texto)->actual = actual;

  objetivo->ant = NULL;
  objetivo->sig = NULL;
  if(lovdog_charfree(&objetivo))
    return error_return("No se pudo liberar el puntero", -1);
  // Reacomodar indices de la tabla
  while (siguient) {
    siguient->indice--;
    siguient = siguient->sig;
  }
  --(*texto)->longitud;
  return 0;
}

/* Busca char en lovdog_texto, r=-1 reinicia iterador, r=0 avanza, r=1 retrocede */
char lovdog_texto_busca(struct lovdog_texto** t, char c, char r){
  if(!*t) return -1;
  if(!(*t)->primer_letra) return -1;
  if(!(*t)->ultima_letra) return -1;
  if((*t)->actual) if(!(*t)->actual->sig) return 1;
  if((*t)->actual) if(!(*t)->actual->ant) return 2;
  // Iniciamos búsqueda
  if(lovdog_texto_circula(t, r)) return 1;
  if((*t)->actual->c==c) return 0;
  while((*t)->actual) {
    if(lovdog_texto_circula(t, r)) return 1;
    if((*t)->actual->c==c) return 0;
  }
  return -2;
}

/* Divide en tokens */
char lovdog_texto_token(struct lovdog_texto** t, char* d){
  if(!*t) return -1;
  if(!(*t)->primer_letra) return -1;
  if(!(*t)->ultima_letra) return -1;
  if(!d) return -2;
 
  struct lovdog_char *actual=NULL;
  size_t
    *indices_ignorados=NULL,
    ocurrencias=0,
    bandera_0=0,
    i=0;

  // Guardamos la posición actual del iterador
  actual=(*t)->actual;
  // Preparamos los valores de inicio
  while((*t)->token_cantidad) (*t)->tokens[--(*t)->token_cantidad]=NULL;
  (*t)->token_cantidad=0;
  if(!(*t)->tokens) free((*t)->tokens);
  if(!(*t)->token_longitud) free((*t)->token_longitud);
  ocurrencias=0;
  // Contar separadores en el texto
  lovdog_texto_circula(t, -1); // Reiniciamos el iterador (!importante)
  while(!lovdog_texto_busca(t, *d, 0)) ++ocurrencias;
  // Si no hay ocurrencias, no hacemos nada
  if(!ocurrencias) { (*t)->actual=actual; return 0; }
  // Guardamos los delimitadores
  (*t)->token_delimitador=(char*)malloc(sizeof(char)*strlen(d)+1);
  strcpy((*t)->token_delimitador, d);
  // Guardamos los índices de los elementos ignorados
  indices_ignorados=(size_t*)malloc(sizeof(size_t)*ocurrencias);
  for(int q=0; q<ocurrencias; q++) indices_ignorados[q]=0;
  ocurrencias=0;
  lovdog_texto_circula(t, -1); // Reiniciamos el iterador (!importante)
  while(!lovdog_texto_busca(t, *d, 0))
    indices_ignorados[ocurrencias++]=(*t)->actual->indice;
  // Conteo de tokens
  // Para esto, es necesario el uso de una bandera, esta bandera
  // actuará como indicador de último estado
  // 0: Separador
  // 1: Token
  bandera_0=0;
  lovdog_texto_circula(t, -1); // Reiniciamos el iterador (!importante)
  while(!lovdog_texto_circula(t, 0)){
    if(en_array(&(*t)->actual->indice, indices_ignorados, ocurrencias, 's'))
      { bandera_0=0; continue; }
    if(!bandera_0) { ++(*t)->token_cantidad; bandera_0=1; }
  }
  // Reserva del espacio necesario para los tokens  encontrados
  (*t)->tokens= (lovdog_char**)malloc(sizeof(lovdog_char*)*(*t)->token_cantidad);
  (*t)->token_longitud= (size_t*)malloc(sizeof(size_t)*(*t)->token_cantidad);
  for(int q=0; q<(*t)->token_cantidad; ++q)
    { (*t)->token_longitud[q]=0; (*t)->tokens[q]=NULL; }
  // Anexado de los tokens
  // Aquí la bandera fungirá como contador de caracteres en el token
  bandera_0=0; i=0;
  lovdog_texto_circula(t, -1); // Reiniciamos el iterador (!importante)
  while(!lovdog_texto_circula(t, 0)){
    if(en_array(&(*t)->actual->indice, indices_ignorados, ocurrencias, 's'))
      { if(bandera_0) ++i; bandera_0=0; continue; }
    if(!bandera_0) { (*t)->tokens[i]=(*t)->actual; bandera_0=1; }
    ++(*t)->token_longitud[i];
  }
  free(indices_ignorados);

  (*t)->actual=actual;
 
  return 0;
}

/* Comparar lovdog_texto */
char lovdog_texto_cmp(struct lovdog_texto* t1, struct lovdog_texto* t2){
  if(!t1 || !t2) return -5;
  if(!t1->primer_letra || !t2->primer_letra) return -6;
  if(!t1->ultima_letra || !t2->ultima_letra) return -7;
  if(!t1->longitud || !t2->longitud) return -8;
  lovdog_char *lc=NULL, *lk=NULL;
  lc=t1->primer_letra;
  lk=t2->primer_letra;
  while(lc && lk){
    if(lc->c > lk->c) return 1;
    if(lc->c < lk->c) return -1;
    lc=lc->sig;
    lk=lk->sig;
  }
  if(lc && !lk) return  1;
  if(lk && !lc) return -1;
  return 0;
}

/* Comparar lovdog_texto tokens */
char lovdog_textok_cmp(struct lovdog_texto* t1, size_t i1, struct lovdog_texto* t2, size_t i2){
  if(!t1 || !t2) return -5;
  if(!t1->primer_letra || !t2->primer_letra) return -6;
  if(!t1->ultima_letra || !t2->ultima_letra) return -6;
  if(!t1->longitud || !t2->longitud) return -6;
  if(!t1->tokens || !t2->tokens) return -7;
  if(!t1->token_longitud || !t2->token_longitud) return -7;
  if(!t1->token_cantidad || !t2->token_cantidad) return -7;
  if(!t1->token_delimitador || !t2->token_delimitador) return -8;
  lovdog_char *lc=NULL, *lk=NULL;
  size_t lc_long=0, lk_long=0;
  lc=t1->tokens[i1]; lc_long=t1->token_longitud[i1];
  lk=t2->tokens[i2]; lk_long=t1->token_longitud[i2];
  while(lc && lk && lc_long && lk_long){
    if(lc->c > lk->c) return 1;
    if(lc->c < lk->c) return -1;
    lc=lc->sig; --lc_long;
    lk=lk->sig; --lk_long;
  }
  if(lc_long > lk_long) return  1;
  if(lc_long < lk_long) return -1;
  return 0;
}

/* Comparar lovdog_texto  con tokens */
char lovdog_texto_token_cmp(struct lovdog_texto* t1, struct lovdog_texto* t2, size_t i2){
  if(!t1 || !t2) return -5;
  if(!t1->primer_letra || !t2->primer_letra) return -6;
  if(!t1->ultima_letra || !t2->ultima_letra) return -6;
  if(!t1->longitud || !t2->longitud) return -6;
  if(!t2->tokens) return -7;
  if(!t2->token_longitud) return -7;
  if(!t2->token_cantidad) return -7;
  if(!t2->token_delimitador) return -8;
  lovdog_char *lc=NULL, *lk=NULL;
  size_t lc_long=0, lk_long=0;
  lc=t1->primer_letra; lc_long=t1->longitud;
  lk=t2->tokens[i2]; lk_long=t2->token_longitud[i2];
  while(lc && lk && lc_long && lk_long){
    if(lc->c > lk->c) return 1;
    if(lc->c < lk->c) return -1;
    lc=lc->sig; --lc_long;
    lk=lk->sig; --lk_long;
  }
  if(lc_long > lk_long) return  1;
  if(lc_long < lk_long) return -1;
  return 0;
}

/* Comparar lovdog_char */
char lovdog_char_cmp(struct lovdog_char* c1, struct lovdog_char* c2){
  if(!c1 || !c2) return -5;
  if(c1->c > c2->c) return  1;
  if(c1->c < c2->c) return -1;
  return 0;
}

/* Copiar lovdog_char, c1 es el apuntador de destino;
 * si puntero=-1, genera una nueva entidad;
 * si puntero=0, comparte punteros;
 * si puntero=1, c1 se queda con los punteros y c2 se queda en null */
char lovdog_texto_cpy( struct lovdog_texto** c1, struct lovdog_texto** c2, char puntero){
  if(!*c2) return -1;
  if(!(*c2)->primer_letra) return -2;
  if(!(*c2)->ultima_letra) return -2;
  if(!(*c2)->longitud) return -2;
  if(puntero>-1){
    if(*c1) lovdog_textofree(c1);
    (*c1)=(*c2);
    if(puntero) *c2=NULL;
    return 0;
  }
  if(!lovdog_texto_cmp(*c1,*c2)){
    if(*c1) lovdog_textofree(c1);
    *c1=lovdog_textoinit();
    if(!*c1) return -3;
    char *aux = lovdog_texto_str(c2);
    lovdog_texto_strappend(c1, aux);
    free(aux);
  }
  if(!(*c2)->tokens) return 0;
  if(!strcmp((*c2)->token_delimitador, (*c1)->token_delimitador)) return 0;
  lovdog_textok_reset(c1);
  lovdog_texto_token(c1, (*c2)->token_delimitador);
  return 0;
}

/* Restablece tokens */
char lovdog_textok_reset(struct lovdog_texto** t){
  if(!(*t)) return -1;
  if(!(*t)->primer_letra) return -2;
  if(!(*t)->ultima_letra) return -2;
  if(!(*t)->longitud) return -2;
  size_t i=0;
  if((*t)->tokens){
    while(i<(*t)->token_cantidad) (*t)->tokens[i++]=NULL;
    free((*t)->tokens);
    (*t)->tokens=NULL;
    free((*t)->token_delimitador);
    (*t)->token_delimitador=NULL;
    free((*t)->token_longitud);
    (*t)->token_longitud=NULL;
    (*t)->token_cantidad=0;
    return 0;
  }
  return 1;
}

/* Compara todos los tokens para verificar si el texto a comparar ya se encuentra en la estructura.
 * El primer elemento es el texto a comparar y el segundo elemento es el texto tokenizado.
 * Esta función retorna la cantidad de tokens iguales a t1.*/
long int lovdog_textok_existe(struct lovdog_texto* t1, struct lovdog_texto* t2){
  if(!t1 || !t2) return -1;
  if(!t2->tokens) return -2;
  size_t i=0, j=0;
  while(i<t2->token_cantidad)
    if(!lovdog_texto_token_cmp(t1, t2, i++)) ++j;
  return j;
}

/* Compara todos los tokens para verificar si el texto a comparar ya se encuentra en la estructura.
 * El primer elemento es el texto a comparar y el segundo elemento es el texto tokenizado.
 * Esta función posiciona el iterador de t1 en el primer token igual a t2 
 * y retorna la posición correspondiente*/
long int lovdog_textok_busca(struct lovdog_texto* t1, struct lovdog_texto* t2){
  if(!t1 || !t2) return -1;
  if(!t2->tokens) return -2;
  size_t i=0;
  while(i<t2->token_cantidad)
    if(!lovdog_texto_token_cmp(t1, t2, i++)) break;
  lovdog_char_en_texto(&t1, i-1);
  return t1->actual->indice;
}

/* Verificar si el sufijo y el prefijo se encuentran en el texto 
 * si el prefijo es NULL, solo se verificará el sufijo, del mismo modo, 
 * si el sufijo es NULL, solo se verificará el prefijo*/
char lovdog_texto_prefijo_sufijo_si(
    struct lovdog_texto** texto,
    const char* prefijo,
    const char* sufijo){
  if(!*texto) return 0;
  if(!(*texto)->primer_letra) return 0;
  if(!(*texto)->ultima_letra) return 0;
  if(!(*texto)->longitud) return 0;
  if(!prefijo && !sufijo) return 0;
  size_t longitud_muestra = 0;
  if(prefijo) longitud_muestra += strlen(prefijo);
  if(sufijo) longitud_muestra += strlen(sufijo);
  if((longitud_muestra+1)>(*texto)->longitud) return 0;
  char salida = 0;
  size_t longitud = 0, i=0;
  struct lovdog_char* lc = (*texto)->primer_letra;
  // Búsqueda del prefijo
  if(prefijo){
    longitud = strlen(prefijo); i=0;
    lc = (*texto)->primer_letra;
    while(lc && longitud>i){
      if(lc->c!=prefijo[i]){ salida=0; break; } 
      lc = lc->sig; ++i; salida=1;
    }
  }
  // Búsqueda del sufijo
  if(sufijo){
    lc = (*texto)->ultima_letra;
    i=strlen(sufijo)-1;
    while(lc && i){
      if(lc->c!=sufijo[i]){ salida=0; break; } 
      lc = lc->ant; --i; salida=1;
    }
  }
  return salida;
}

/* Verificar si el sufijo y el prefijo se encuentran en el token 
 * si el prefijo es NULL, solo se verificará el sufijo, del mismo modo, 
 * si el sufijo es NULL, solo se verificará el prefijo*/
char lovdog_textok_prefijo_sufijo_si(
    struct lovdog_texto** texto,
    size_t token,
    const char* prefijo,
    const char* sufijo
){
  if(!*texto) return 0;
  if(!(*texto)->primer_letra) return 0;
  if(!(*texto)->ultima_letra) return 0;
  if(!(*texto)->longitud) return 0;
  if(!(*texto)->tokens) return 0;
  if(token>=(*texto)->token_cantidad) return 0;
  if(!prefijo && !sufijo) return 0;
  size_t longitud_muestra = 0;
  if(prefijo) longitud_muestra += strlen(prefijo);
  if(sufijo) longitud_muestra += strlen(sufijo);
  if((longitud_muestra+1)>(*texto)->token_longitud[token]) return 0;
  char salida = 0;
  size_t longi = 0, longj = 0, i=0, j=0;
  struct lovdog_char* lc = NULL; 
  // Búsqueda del prefijo
  if(prefijo){
    longi = strlen(prefijo); i=0;
    longj = (*texto)->token_longitud[token]; j=0;
    lc = (*texto)->tokens[token];
    while(longj>j && longi>i){
      if(lc->c!=prefijo[i]){ salida=0; break; } 
      lc = lc->sig; ++i; salida=1; ++j;
    }
  }
  // Búsqueda del sufijo
  if(sufijo){
    lc = (*texto)->tokens[token];
    longi = strlen(sufijo);
    longj = (*texto)->token_longitud[token];
    i=0; j=longi-longj;
    while(j){ lc=lc->sig; --j; ++i; }
    while(i<longi && j<longj){
      if(lc->c!=sufijo[i]){ salida=0; break; } 
      lc = lc->ant; ++i; salida=1; ++j;
    }
  }
  return salida;
}

/* Regresa una cadena sin el prefijo y sin el sufijo especificados */
char* lovdog_texto_prefijo_sufijo_sin(
    struct lovdog_texto** texto,
    const char* prefijo,
    const char* sufijo
){
  if(!*texto) return NULL;
  if(!(*texto)->primer_letra) return NULL;
  if(!(*texto)->ultima_letra) return NULL;
  if(!(*texto)->longitud) return NULL;
  if(!prefijo && !sufijo) return NULL;
  // Se verifica si existe el prefijo y el sufijo
  char prefix_suffix=0;
  if(lovdog_texto_prefijo_sufijo_si(texto, prefijo, NULL)) prefix_suffix+=1;
  if(lovdog_texto_prefijo_sufijo_si(texto, NULL, sufijo)) prefix_suffix+=2;
  if(!prefix_suffix) return NULL;
  // Si los afijos existen, se obtiene la cantidad de texto
  // que corresponde a estos.
  size_t longitud_afijos = 0, longitud_lexema = 0;
  switch (prefix_suffix){
    case 1: longitud_afijos = strlen(prefijo); break;
    case 2: longitud_afijos = strlen(sufijo); break;
    case 3: longitud_afijos = strlen(prefijo)+strlen(sufijo); break;
  }
  // Cantidad de texto que corresponde al texto sin afijos
  longitud_lexema = (*texto)->longitud - longitud_afijos;
  // Preparamos iteradores
  /// Guardamos posición actual del iterador del texto
  struct lovdog_char* actual = (*texto)->actual;
  // Ubicamos al iterador del texto en la primera letra del lexema
  if(prefix_suffix!=2) lovdog_char_en_texto(texto, strlen(prefijo));
  else lovdog_char_en_texto(texto, 0);
  struct lovdog_char* iterador = (*texto)->actual;
  (*texto)->actual=actual; // Regresamos el iterador a su posición original
  // Creamos la cadena
  char* salida = (char*)malloc(sizeof(char)*longitud_lexema+1);
  strcpy(salida, "");
  for(int i=0; i<longitud_lexema; ++i){
    salida[i] = iterador->c;
    iterador = iterador->sig;
  }
  return salida;
}

/* Regresa una cadena sin el prefijo y sin el sufijo especificados versión token */
char* lovdog_textok_prefijo_sufijo_sin(
    struct lovdog_texto** texto,
    size_t token, const char* prefijo,
    const char* sufijo
){
  if(!*texto) return NULL;
  if(!(*texto)->primer_letra) return NULL;
  if(!(*texto)->ultima_letra) return NULL;
  if(!(*texto)->longitud) return NULL;
  if(!(*texto)->token_cantidad) return NULL;
  if(!(*texto)->tokens) return NULL;
  if((*texto)->token_cantidad<=token) return NULL;
  if(!prefijo && !sufijo) return NULL;
  // Se verifica si existe el prefijo y el sufijo
  char prefix_suffix=0;
  if(lovdog_textok_prefijo_sufijo_si(texto, token, prefijo, NULL)) prefix_suffix+=1;
  if(lovdog_textok_prefijo_sufijo_si(texto, token, NULL, sufijo)) prefix_suffix+=2;
  if(!prefix_suffix) return NULL;
  // Si los afijos existen, se obtiene la cantidad de texto
  // que corresponde a estos.
  size_t longitud_afijos = 0, longitud_lexema = 0;
  switch (prefix_suffix){
    case 1: longitud_afijos = strlen(prefijo); break;
    case 2: longitud_afijos = strlen(sufijo); break;
    case 3: longitud_afijos = strlen(prefijo)+strlen(sufijo); break;
  }
  // Cantidad de texto que corresponde al texto sin afijos
  longitud_lexema = (*texto)->token_longitud[token] - longitud_afijos;
  // Preparamos iteradores
  /// Guardamos posición actual del iterador del texto
  struct lovdog_char* actual = (*texto)->actual;
  // Ubicamos al iterador del texto en la primera letra del lexema
  if(prefix_suffix!=2) lovdog_char_en_texto(texto, (*texto)->token_longitud[token] + strlen(prefijo));
  else lovdog_char_en_texto(texto, (*texto)->token_longitud[token]);
  struct lovdog_char* iterador = (*texto)->actual;
  (*texto)->actual=actual; // Regresamos el iterador a su posición original
  // Creamos la cadena
  char* salida = (char*)malloc(sizeof(char)*longitud_lexema+1);
  strcpy(salida, "");
  for(int i=0; i<longitud_lexema; ++i){
    salida[i] = iterador->c;
    iterador = iterador->sig;
  }
  return salida;
}

/* Agrega un prefijo y un sufijo a lovdog_texto */
char lovdog_texto_prefijo_sufijo_add(
    struct lovdog_texto** texto,
    const char* prefijo,
    const char* sufijo
){
  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -2;
  if(!(*texto)->ultima_letra) return -2;
  if(!(*texto)->longitud) return -2;
  if(prefijo) lovdog_texto_strinsert(texto,(char*)prefijo, 0);
  if(sufijo) lovdog_texto_strinsert(texto,(char*)sufijo, (*texto)->ultima_letra->indice);
  return 0;
}

/* Agrega un prefijo y un sufijo a lovdog_texto versión token */
char lovdog_textok_prefijo_sufijo_add(
    struct lovdog_texto** texto,
    size_t token,
    const char* prefijo,
    const char* sufijo
){
  if(!*texto) return -1;
  if(!(*texto)->primer_letra) return -2;
  if(!(*texto)->ultima_letra) return -2;
  if(!(*texto)->longitud) return -2;
  if(!(*texto)->token_cantidad) return -3;
  if(!(*texto)->tokens) return -3;
  if((*texto)->token_cantidad<=token) return -3;
  if(prefijo)
    lovdog_texto_strinsert(texto,(char*)prefijo, (*texto)->tokens[token]->indice);
  if(sufijo)  
    lovdog_texto_strinsert(texto,(char*)sufijo, (*texto)->tokens[token]->indice+(*texto)->token_longitud[token]-1);
  return 0;
}

/* Lovdog textok a texto */
lovdog_texto* lovdog_textok_texto(struct lovdog_texto** texto, size_t token){
  if(!*texto) return NULL;
  if(!(*texto)->primer_letra) return NULL;
  if(!(*texto)->ultima_letra) return NULL;
  if(!(*texto)->longitud) return NULL;
  if(!(*texto)->token_cantidad) return NULL;
  if(!(*texto)->tokens) return NULL;
  if((*texto)->token_cantidad<=token) return NULL;
  struct lovdog_texto* salida = lovdog_textoinit();
  struct lovdog_char* iterador = (*texto)->tokens[token];
  size_t longitud=(*texto)->token_longitud[token];
  size_t i=0;
  if(!salida) return NULL;
  while(iterador && i<longitud){
    lovdog_texto_charappend(&salida, iterador->c);
    iterador=iterador->sig; ++i;
  }
  return salida;
}
