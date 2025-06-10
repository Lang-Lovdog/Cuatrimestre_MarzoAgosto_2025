#ifndef __LOVDOG_AG__
#define __LOVDOG_AG__
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>

namespace lovdog {

class Individuo {
public:
  Individuo();
  Individuo(size_t* nucleo, size_t n);
  Individuo(Individuo& ente);
 ~Individuo();
  /*--------------*/
  size_t* cromosoma;
  size_t  tamanno;
  float   aptitud;
  float   coste;
  bool    vivo;
  /*--------------*/

  /*--------------*/
  friend std::ostream& operator<< (std::ostream& os, const Individuo& ente);
  /*--------------*/
};

class lovdogListaNodos{
public:
  lovdogListaNodos();
  lovdogListaNodos(const char* archivoCSV);
  lovdogListaNodos(const char* archivoCSV, const char tipo);
  lovdogListaNodos(const lovdogListaNodos& lln);
  lovdogListaNodos(lovdogListaNodos& lln);
  ~lovdogListaNodos();
  /*--------------*/
  bool        imprimeMatrizAdyacencias(void);
  float*      celdaMatrizAdyacencias(size_t x, size_t y);
  float       distanciaEuclideana(size_t indexA, size_t indexB);
  size_t*     rutaAleatoria(void);
  size_t      cardinalidad(void) const;
  size_t      dimensionPorNodo(void) const;
  void        leeCSV(std::string nombreArchivo, const char headers_indexes);
  float*      nodoEn(size_t idx) const;
  float       Xa(size_t idx) const;
  /*--------------*/
  static const char
    CSV_SOLO_DATOS = 0b00,
    CSV_CABECERA = 0b01,
    CSV_INDICES = 0b10,
    CSV_CABECERA_INDICES = 0b11,
    CSV_INDICE_TAG = 0b01,
    CSV_HEAD_TAG = 0b10,
    CSV_TAG = 0b11
  ;
  /*--------------*/
  float  operator [](size_t idx) const;
  float* operator ()(size_t idx) const;
  /*--------------*/
private:
  float*  x;
  float*  Adyacencias;
  size_t  tamanno;
  size_t  dimensiones;
  char*   i;
  char    tagged;
  /*--------------*/
  void inicializaX(void);
  bool creaMatrizAdyacencias(void);
  /*--------------*/
  friend std::ostream& operator << (std::ostream& os, const lovdogListaNodos& grafo);
  /*--------------*/
};

typedef lovdogListaNodos Grafo;
typedef unsigned int uint_t;

class Geneticos {
public:
  // Constructors and destructors
  Geneticos();
  Geneticos(Geneticos& poblacion);
  Geneticos(const char* archivo);
  Geneticos(const char* archivo, const char formato);
  Geneticos(const lovdogListaNodos* listaNodos);
 ~Geneticos();

  // Atributos
  size_t numGeneraciones;
  size_t cantidadIndividuos;
  uint_t tipoCruce;
  uint_t tipoIndividuo;


  // Métodos de la clase
  //// Mutación
  void define(const char campo, const void* valor);
  void estableceEvaluador(float (*evalFn)(size_t*,Grafo*));

  //// Acceso
  Individuo elIndividuo(size_t indice);
  Individuo elMejorIndividuo(void);
  bool imprimeResultados(void);

  //// Auxiliares
  bool guardaNodos(const char* archivo, const char formato);
  void evaluar(void);

  //// Estáticos
  const static char 
    // Opciones de población
    AG_NUM_GENERACIONES=0,
    AG_TIPO_CRUCE=1,
    AG_CANTIDAD_INDIVIDUOS=2,
    AG_CANTIDAD_CRUCES=3,
    AG_TIPO_INDIVIDUO=4
  ;
  const static uint_t
    // Opciones de tipo de cruce
    OX_CROSS = 0,
    PERMLANG_CROSS = 1,
    // Opciones de tipo de individuo
    INDIVIDUO_BINARIO=0,
    INDIVIDUO_PERMUTACION=1
  ;

private:
  // Atributos
  lovdogListaNodos* nodos;
  Individuo*        poblacion;
  Individuo*        nPoblacion;
  size_t*           posiblesval;
  size_t            mejorIndividuo;
  size_t            crucesPorPoblacion;
  float*            probabilidadIndividuo;

  // Métodos
  //// Control de población
  bool creaPoblacion(void);
  bool creaIndividuo(Individuo* nuevoIndividuo);
  bool iniciaPoblacion(void);
  bool mejorRendimientoEncuentra(void);

  //// Funciones de individuo
  static void cruceOx(const Individuo& padre, const Individuo& madre, Individuo* hijos);
  static void crucePermLang(const Individuo& padre, const Individuo& madre, Individuo* hijos);
  void depredadorMatador(void);

  //// Contenedores genéricos
  float (*aptitudEvalFn)(size_t*,Grafo*);
  void  (*generador)(Individuo*, size_t);
  void  (*cruce)(const Individuo&, const Individuo&, Individuo*);

  //// Funciones de configuración
  void cambiaTipoCruce(void);
  void cambiaTipoGenerador(void);

  //// Auxiliares
  static void generadorPermutado(Individuo*, size_t);
  static void generadorBinario(Individuo*, size_t);
  float  celdaAleatoriaRestringida(void);

  //// Amigos
  friend std::ostream& operator<< (std::ostream& os, const Geneticos& g);
};
 
}

#endif
