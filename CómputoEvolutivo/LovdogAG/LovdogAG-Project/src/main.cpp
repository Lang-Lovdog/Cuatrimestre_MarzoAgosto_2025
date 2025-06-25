#include "lovdogAG.hxx"
#include <iostream>

void mainDemoGaafo(void);
void mainAGLOVDOG(const char* Archivo);
void mainAGRestrictedLOVDOG(const char* Archivo);

int main (int argc, char** argv){
  //mainDemoGaafo();
  //if(argc<2){ std::cerr << "No archivos para leer" << std::endl; return 1; }
  if(argc<3){
    std::cerr << "Uso correcto\n"
              << argv[0] << " "
              << "<r | n> "
              << "archivo.csv\n"
              << "Con la opción r, se deberá proporcionar una matriz de adyacencias\n"
              << "Con la opción n, se deberá proporcionar una tabla de puntos (nodos)\n"
              << "Inicialmente, el programa supone que el archivo csv tiene una cabecera" 
              << "y una columna de índices. Modifíque las macros en caso de ser necesario."
              << std::endl;
    return 1; }
  switch(**(argv+1)){
    case 'r':
      mainAGRestrictedLOVDOG(*(argv+2));
      break;
    case 'n':
      mainAGLOVDOG(*(argv+2));
      break;
  }
  return 0;
}

void mainAGRestrictedLOVDOG(const char* Archivo){
  lovdog::Geneticos M10(Archivo, lovdog::Grafo::CSV_CABECERA_INDICES | lovdog::Grafo::CSV_ADYACENCIAS_INPUT);
  M10.define(lovdog::Geneticos::AG_TIPO_SELECCION,      lovdog::Geneticos::SELECCION_ELITISMO);
  M10.define(lovdog::Geneticos::AG_TIPO_INDIVIDUO,      lovdog::Geneticos::INDIVIDUO_PERMUTACION);
  M10.define(lovdog::Geneticos::AG_TIPO_CRUCE,          lovdog::Geneticos::CRUCE_OX_CROSS);
  M10.define(lovdog::Geneticos::AG_VERBOSITY,           lovdog::Geneticos::VERBOSITY_RESUMEN);
  M10.define(lovdog::Geneticos::AG_CANTIDAD_CRUCES,     (size_t)4);
  M10.define(lovdog::Geneticos::AG_NUM_GENERACIONES,    (size_t)500);
  M10.define(lovdog::Geneticos::AG_CANTIDAD_INDIVIDUOS, (size_t)120);
  M10.define(lovdog::Geneticos::AG_CANTIDAD_MUTACIONES, (size_t)5);
  M10.estableceEvaluador(lovdog::Geneticos::TSPEvaluador);
  //std::cout << M10.grafo();
  M10.ejecuta(lovdog::Geneticos::AG_INICIA);
  std::cerr << M10;
  std::cout << "TSP Restricted terminated";
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
  //TSP100.ejecuta(lovdog::Geneticos::AG_INICIA);
  //std::cerr << TSP100;
  //std::cout << "TSP terminated";
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
