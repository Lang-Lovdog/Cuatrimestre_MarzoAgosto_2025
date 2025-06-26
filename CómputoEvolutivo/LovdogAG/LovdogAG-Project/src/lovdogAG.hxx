#ifndef __LOVDOG_AG__
#define __LOVDOG_AG__
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

#define LDP << std::setprecision(19)

namespace lovdog {

class Individuo {
public:
  Individuo();
  Individuo(size_t* nucleo, size_t n);
  Individuo(Individuo& ente);
  Individuo(size_t n, bool);
 ~Individuo();
  /*--------------*/
  size_t* cromosoma;
  size_t  tamanno;
  float   aptitud;
  float   coste;
  bool    vivo;
  /*--------------*/
  static bool existe(const Individuo* ente, size_t* nucleotido);
  /*--------------*/
  friend std::ostream& operator<< (std::ostream& os, const Individuo& ente);
  /*--------------*/
  Individuo& operator=(const Individuo& ente);
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
  bool        imprimeMatrizAdyacencias(void) const;
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
    CSV_INDICE_TAG = 0b0100,
    CSV_HEAD_TAG = 0b1000,
    CSV_TAG = 0b1100,
    CSV_ADYACENCIAS_INPUT = 0b10000
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
  bool    soloAdyacencias;
  /*--------------*/
  void inicializaX(void);
  void inicializaM(void);
  bool creaMatrizAdyacencias(void);
  bool creaMatrizAdyacencias(std::string nombreArchivo, const char headers_indexes);
  void penalizacionMatriz(void);
  /*--------------*/
  friend std::ostream& operator << (std::ostream& os, const lovdogListaNodos& grafo);
  /*--------------*/
  lovdogListaNodos& operator=(const lovdogListaNodos& lln);
  /*--------------*/
};

typedef lovdogListaNodos Grafo;
typedef unsigned int  uint_t;
typedef unsigned char uchar_t;

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
  uint_t tipoSeleccion;
  uint_t tipoReemplazo;
  uint_t tipoIndividuo;
  uint_t mutacionesPorGeneracion;
  size_t generacionActual;
  char   verbose;

  //// Estáticos
  const static uchar_t
    // Opciones de población
    AG_NUM_GENERACIONES=0,
    AG_TIPO_CRUCE=1,
    AG_CANTIDAD_MUTACIONES=2,
    AG_CANTIDAD_INDIVIDUOS=3,
    AG_CANTIDAD_CRUCES=4,
    AG_TIPO_INDIVIDUO=5,
    AG_TIPO_SELECCION=6,
    AG_TIPO_REEMPLAZO=7,
    AG_VERBOSITY=8,
    /*--------------*/
    AG_PRIMERA_GENERACION=0,
    AG_CRUCE=1,
    AG_DEPREDADOR=2,
    AG_REEMPLAZO=3,
    AG_EVALUACION=4,
    AG_SELECCION=5,
    AG_MUTACION=6,
    AG_INICIA=250
  ;
  const static uint_t
    // Opciones de tipo de cruce
    CRUCE_OX_CROSS = 0,
    CRUCE_PERMLANG_CROSS = 1,
    // Opciones de tipo de reemplazo
    REEMPLAZO_ALEATORIO = 0,
    REEMPLAZO_ELITISMO = 1,
    // Opciones de tipo de individuo
    INDIVIDUO_BINARIO=0,
    INDIVIDUO_PERMUTACION=1,
    // Opciones de tipo de seleccion
    SELECCION_ELITISMO=0,
    // Verbosity level
    VERBOSITY_NADA=0,
    VERBOSITY_GENERACIONES=1,
    VERBOSITY_RESUMEN=2
  ;


  // Métodos de la clase
  //// Mutación
  void define(const char campo, uint_t valor);
  void define(const char campo, size_t valor);
  void estableceEvaluador(float (*evalFn)(size_t*,Grafo*));

  //// Acceso
  Individuo elIndividuo(size_t indice);
  Individuo elMejorIndividuo(void);
  bool imprimeResultados(void);
  lovdogListaNodos grafo(void) const;

  //// Auxiliares
  bool guardaNodos(const char* archivo, const char formato);
  void ejecuta(uchar_t evento);

  // Funciones de evaluación predefinidas
  static float TSPEvaluador(size_t*,Grafo*);

private:
  // Atributos
  lovdogListaNodos* nodos;
  Individuo         mejorhst;
  Individuo*        poblacion;
  Individuo*        nPoblacion;
  size_t*           posiblesval;
  size_t            mejorIndividuo;
  float             mejorRendimiento;
  size_t            crucesPorPoblacion;
  float*            probabilidadIndividuo;
  float             mejorRendimientoHistorico;
  bool              mejorRendimientoHistoricoUsado;

  // Métodos
  //// Control de población
  void temporadaApareamiento(void);
  void depredadorMatadorAtak(Individuo*);
  static void choqueGeneracionalElitista(Individuo*,Individuo*, size_t, size_t);
  //static void choqueGeneracionalAleatorio(Individuo*,Individuo*,size_t, size_t);
  static void choqueGeneracional(Individuo*,Individuo*,size_t, size_t);

  //// Funciones de individuo
  static void cruceOx(const Individuo* padre, const Individuo* madre, Individuo* hijos, size_t ndh);
  static void crucePermLang(const Individuo* padre, const Individuo* madre, Individuo* hijos, size_t ndh);
  static void reiniciadorDIndividuo(const Individuo*, size_t);
  static void reiniciadorDIndividuo(Individuo&);
  void depredadorMatador(void);
  void evaluar(void);

  //// Contenedores genéricos
  float (*aptitudEvalFn)(size_t*,Grafo*);
  void  (*generador)(Individuo**, size_t, const Grafo*);
  void  (*cruce)(const Individuo*, const Individuo*, Individuo*, size_t);
  void  (*seleccion)(const Individuo*, size_t*, size_t*);
  void  (*mutacion)(const Individuo*, size_t);
  void  (*reemplazo)(Individuo* poblacion, Individuo* hijos, size_t n, size_t h);

  //// Funciones de configuración
  void cambiaTipoCruce(void);
  void cambiaTipoGenerador(void);
  void cambiaTipoSeleccion(void);
  void cambiaTipoReemplazo(void);

  //// Auxiliares
  static void generadorPermutado(Individuo**, size_t, const Grafo*);
  static void generadorBinario(Individuo**, size_t, const Grafo*);
  static void elitismo(const Individuo* poblacion, size_t* seleccionados, size_t* opciones);
  float  celdaAleatoriaRestringida(void);

  // Ciclos de evolución
  void cicloSimple(void);

  //// Amigos
  friend std::ostream& operator<< (std::ostream& os, const Geneticos& g);
};
 
}

#endif
