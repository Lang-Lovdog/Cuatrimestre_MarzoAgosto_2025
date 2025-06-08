#include "lovdogAG.hxx"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace lovdog;

bool Poblacion::guardaNodos(const char* archivo, const char* formato){
  return true;
}

bool Poblacion::inicializaVectorValores(){
  return true;
}

bool Poblacion::creaPoblacion(void){
  size_t idx;
  this->poblacion = nullptr;

  this->poblacion = new individuo[this->cantidadIndividuos];
  if(this->poblacion == nullptr) return false;

  idx=0;
  while(idx < this->cantidadIndividuos){
    (this->poblacion+idx)->cromosoma = nullptr;
    (this->poblacion+idx)->cromosoma = new size_t[this->nodos.cardinalidad()];
    if((this->poblacion+idx)->cromosoma == nullptr){
      while(idx>0){
        --idx;
        delete[] (this->poblacion+idx)->cromosoma;
      }
      delete[] poblacion;
      return false;
    }
    (this->poblacion+idx)->coste=0;
    (this->poblacion+idx)->precision=0;
    ++idx;
  }
  return true;
}

bool Poblacion::creaIndividuo(individuo* nuevoIndividuo){
  // Se asume que el espacio necesario para el individuo ya ha sido asignado
  size_t
    rndIdx,
    genIdx;

  bool* usedFlag;
  usedFlag = new bool[this->nodos.cardinalidad()];

  if(nuevoIndividuo->cromosoma == nullptr ) nuevoIndividuo->cromosoma = new size_t[this->nodos.cardinalidad()];

  genIdx=0; while(genIdx < this->nodos.cardinalidad()) *(usedFlag+(genIdx++))=false;

  genIdx=rndIdx=0;
  while(genIdx < this->nodos.cardinalidad()){
    if(*(usedFlag+rndIdx)) nuevoIndividuo->cromosoma[genIdx];
  }
  return true;
}



// Lovdog Lista Nodos
lovdogListaNodos::lovdogListaNodos(){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
  this->leeCSV(archivoCSV, CSV_SOLO_DATOS);
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV, const char tipo){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
  this->leeCSV(archivoCSV, tipo);
}

lovdogListaNodos::lovdogListaNodos(lovdogListaNodos& lln){
}

lovdogListaNodos::~lovdogListaNodos(){
  delete[] this->x;
  delete[] this->Adyacencias;
}

bool lovdogListaNodos::creaMatrizAdyacencias(void){
  float* matriz;
  size_t  auxIdx, auxIdy, mSize, *n ;

  n=&this->tamanno; mSize=(*n)*(*n);
  matriz = new float[mSize]; if(matriz==nullptr) return false;

  auxIdx = auxIdy = 0;
  while(auxIdx < mSize){
    while(auxIdx < mSize){
      *(matriz+auxIdx+(*n*auxIdy)) = this->distanciaEucliedana(auxIdx, auxIdy);
      ++auxIdx;
    }
    ++auxIdy;
  }

  if(this->Adyacencias != nullptr) delete[] this->Adyacencias;

  this->Adyacencias = matriz; matriz = nullptr;

  return true;
}

float lovdogListaNodos::distanciaEucliedana(size_t indexA, size_t indexB){
  if( indexA > this->tamanno ) indexA = indexA % this->tamanno;
  if( indexB > this->tamanno ) indexB = indexB % this->tamanno;
  indexA *= this->dimensiones;
  indexB *= this->dimensiones;
  float* x = this->x;
  return sqrtf(
      powf(*(x+indexB)-*(x+indexA), 2) +
      powf(*(x+1+indexB)-*(x+1+indexA), 2)
  );
}

float* lovdogListaNodos::celdaMatrizAdyacencias(size_t x, size_t y){
  x=x%this->tamanno; // Índice circular
  y=y%this->tamanno; // Índice circular
  return (this->Adyacencias+x+(y*this->tamanno));
}

void lovdogListaNodos::leeCSV(std::string nombreArchivo, const char headers_indexes){
  // Se asume que el texto no tiene cabecera y es únicamente una lista de
  // puntos numéricos
  std::fstream csvEnCuestion;
  std::stringstream token;
  std::string linea;
  std::vector<float> valores;
  char idx; size_t xidx;

  csvEnCuestion.open(nombreArchivo, std::ios::in);
  if(!csvEnCuestion.is_open()){ std::cout << "Error al leer CSV"; return; }
  this->tamanno = std::count(
      std::istreambuf_iterator<char>(csvEnCuestion),
      std::istreambuf_iterator<char>(), '\n'
  );
  csvEnCuestion.clear();
  csvEnCuestion.seekg(0);

  std::getline(csvEnCuestion, linea);
  
  token = std::stringstream(linea);
  
  std::cout << linea << std::endl;

  if(headers_indexes & CSV_CABECERA)
    while(std::getline(token, linea, ','))
      ++this->dimensiones;
  else{
    while(std::getline(token, linea, ','))
      if(!linea.empty() )
        valores.push_back(atof(linea.c_str()));
    this->dimensiones=valores.size();
  }
  if(headers_indexes & CSV_INDICES){
    --this->dimensiones;
    this->x = new float[(this->dimensiones)*(this->tamanno)];
    x[0] = valores.at(0); x[1] = valores.at(1); valores.clear();
    xidx=0;
    while(std::getline(csvEnCuestion,linea)){
      token << linea; idx=0;
      while(std::getline(token,linea,',')) if((idx++)%this->dimensiones) *(this->x+(xidx++))=atof(linea.c_str());
    }
  }else{
    xidx=0;
    while(std::getline(csvEnCuestion,linea)){
      token << linea;
      while(std::getline(token,linea,',')) *(this->x+(xidx++))=atof(linea.c_str());
    }
  }
  csvEnCuestion.close();
}

size_t lovdogListaNodos::dimensionPorNodo(void) { return dimensiones; }
size_t lovdogListaNodos::cardinalidad(void){ return tamanno; }
