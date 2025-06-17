#include "lovdogAG.hxx"
#include <iostream>

void mainDemoGaafo(void);
void mainAGLOVDOG(const char* Archivo);
void mainAGRestrictedLOVDOG(const char* Archivo);

int main (int argc, char** argv){
  //mainDemoGaafo();
  if(argc<2){ std::cerr << "No archivos para leer" << std::endl; return 1; }
  //mainAGLOVDOG(argv[1]);
  mainAGRestrictedLOVDOG(NULL);
  return 0;
}

void mainAGRestrictedLOVDOG(const char* Archivo){
  lovdog::Geneticos TSP100(Archivo, lovdog::Grafo::CSV_CABECERA_INDICES | lovdog::Grafo::CSV_ADYACENCIAS_INPUT);
  TSP100.define(lovdog::Geneticos::AG_TIPO_SELECCION,      lovdog::Geneticos::SELECCION_ELITISMO);
  TSP100.define(lovdog::Geneticos::AG_TIPO_INDIVIDUO,      lovdog::Geneticos::INDIVIDUO_PERMUTACION);
  TSP100.define(lovdog::Geneticos::AG_TIPO_CRUCE,          lovdog::Geneticos::CRUCE_OX_CROSS);
  TSP100.define(lovdog::Geneticos::AG_VERBOSITY,           lovdog::Geneticos::VERBOSITY_RESUMEN);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_CRUCES,     (size_t)4);
  TSP100.define(lovdog::Geneticos::AG_NUM_GENERACIONES,    (size_t)500);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_INDIVIDUOS, (size_t)120);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_MUTACIONES, (size_t)5);
  TSP100.estableceEvaluador(lovdog::Geneticos::TSPEvaluador);
  TSP100.ejecuta(lovdog::Geneticos::AG_INICIA);
  std::cerr << TSP100;
}

void mainAGLOVDOG(const char* Archivo){
  lovdog::Geneticos TSP100(Archivo,lovdog::Grafo::CSV_CABECERA_INDICES);
  TSP100.define(lovdog::Geneticos::AG_TIPO_SELECCION,      lovdog::Geneticos::SELECCION_ELITISMO);
  TSP100.define(lovdog::Geneticos::AG_TIPO_INDIVIDUO,      lovdog::Geneticos::INDIVIDUO_PERMUTACION);
  TSP100.define(lovdog::Geneticos::AG_TIPO_CRUCE,          lovdog::Geneticos::CRUCE_OX_CROSS);
  TSP100.define(lovdog::Geneticos::AG_VERBOSITY,           lovdog::Geneticos::VERBOSITY_RESUMEN);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_CRUCES,     (size_t)4);
  TSP100.define(lovdog::Geneticos::AG_NUM_GENERACIONES,    (size_t)500);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_INDIVIDUOS, (size_t)120);
  TSP100.define(lovdog::Geneticos::AG_CANTIDAD_MUTACIONES, (size_t)5);
  TSP100.estableceEvaluador(lovdog::Geneticos::TSPEvaluador);
  TSP100.ejecuta(lovdog::Geneticos::AG_INICIA);
  std::cerr << TSP100;
}

void mainDemoGaafo(void){
  lovdog::Grafo grafo("../tsp_100_cities.csv", lovdog::Grafo::CSV_CABECERA_INDICES);
  std::cout 
    << "Grafo con " << grafo.cardinalidad()
    << " nodos de " << grafo.dimensionPorNodo() 
    << " dimensiones" << std::endl
    << grafo
  ;
  grafo.imprimeMatrizAdyacencias();
}
