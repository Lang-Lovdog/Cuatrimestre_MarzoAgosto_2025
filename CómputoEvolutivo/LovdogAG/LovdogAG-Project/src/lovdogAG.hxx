#ifndef __LOVDOG_AG__
#define __LOVDOG_AG__
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <string>

namespace lovdog {

struct individuo {
  size_t* cromosoma;
  float   precision;
  float   coste;
};

class lovdogListaNodos{
public:
  lovdogListaNodos();
  lovdogListaNodos(const char* archivoCSV);
  lovdogListaNodos(const char* archivoCSV, const char tipo);
  lovdogListaNodos(lovdogListaNodos& lln);
  ~lovdogListaNodos();
  /*--------------*/
  bool    creaMatrizAdyacencias(void);
  float*  celdaMatrizAdyacencias(size_t x, size_t y);
  float   distanciaEucliedana(size_t indexA, size_t indexB);
  size_t* rutaAleatoria(void);
  size_t  cardinalidad(void);
  size_t  dimensionPorNodo(void);
  void    leeCSV(std::string nombreArchivo, const char headers_indexes);
  /*--------------*/
  static const char
    CSV_SOLO_DATOS = 0b00,
    CSV_CABECERA = 0b01,
    CSV_INDICES = 0b10,
    CSV_CABECERA_INDICES = 0b11
  ;

  /*--------------*/
private:
  float*  x;
  float*  Adyacencias;
  size_t  tamanno;
  size_t  dimensiones;
  /*--------------*/
};

typedef lovdogListaNodos Grafo;
typedef struct individuo individuo;
typedef unsigned int uint_t;

class Poblacion {
public:
  // Constructors and destructors
  Poblacion();
  Poblacion(Poblacion& poblacion);
  Poblacion(const char* archivo, const char* formato);
  Poblacion(const void* listaNodos);
 ~Poblacion();

  // Atributos
  size_t numGeneraciones;
  size_t cantidadIndividuos;
  uint_t tipoCruce;


  // Métodos de la clase
  //// Mutación
  bool define(const char* campo, const void* valor);

  //// Acceso
  individuo elIndividuo(size_t indice);
  individuo elMejorIndividuo(void);
  bool imprimeResultados(void);

  //// Auxiliares
  bool guardaNodos(const char* archivo, const char* formato);

private:
  // Atributos
  individuo* poblacion;
  size_t     mejorIndividuo;
  size_t*    vectorValores;
  lovdogListaNodos nodos;

  // Métodos
  //// Control de población
  bool creaPoblacion(void);
  bool creaIndividuo(individuo* nuevoIndividuo);
  bool iniciaPoblacion(void);
  bool mejorRendimientoEncuentra(void);

  //// Funciones de cruce
  bool cruceOx(const individuo& padre, const individuo& madre, const individuo hijo);
  bool crucePermLang(const individuo& padre, const individuo& madre, const individuo hijo);
  bool depredadorMatador(void);

  //// Auxiliares
  bool   inicializaVectorValores(void);
  float  celdaAleatoriaRestringida();
};
 
}

#endif
