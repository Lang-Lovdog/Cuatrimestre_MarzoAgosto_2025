#include "lovdogAG.hxx"
#include <iostream>

int main (void){
  lovdog::Grafo grafo("../tsp_100_cities.csv", lovdog::CSV_CABECERA);
  std::cout << "Grafo con " << grafo.cardinalidad() << " nodos de " << grafo.dimensionPorNodo() << " dimensiones" << std::endl;
  return 0;
}
