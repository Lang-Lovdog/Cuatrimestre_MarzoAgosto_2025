#include "lovdogAG.hxx"
#include <iostream>

int main (void){
  lovdog::Grafo grafo("../tsp_100_cities.csv", lovdog::Grafo::CSV_CABECERA_INDICES);
  std::cout 
    << "Grafo con " << grafo.cardinalidad()
    << " nodos de " << grafo.dimensionPorNodo() 
    << " dimensiones" << std::endl
    << grafo
  ;
  grafo.imprimeMatrizAdyacencias();
  return 0;
}
