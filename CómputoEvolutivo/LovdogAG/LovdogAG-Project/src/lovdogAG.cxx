#include "lovdogAG.hxx"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace lovdog{

// Lovdog Individuo
Individuo::Individuo(){
  this->tamanno=0;
  this->aptitud=0;
  this->coste=0;
  this->vivo=true;
  this->cromosoma=nullptr;
}

Individuo::Individuo(size_t* nucleo, size_t n){
  size_t idx;
  idx=0;
  this->aptitud=0;
  this->coste=0;
  this->tamanno=n;
  this->vivo=true;
  this->cromosoma= new size_t[n];
  while(idx<n){ *(this->cromosoma+idx) = *(nucleo+idx); ++idx; }
}

Individuo::Individuo(Individuo& ente){
  if(!(ente.tamanno && ente.cromosoma)) return;
  size_t idx; idx=0;
  this->tamanno=ente.tamanno;
  this->aptitud=ente.aptitud;
  this->coste=ente.coste;
  this->vivo=ente.vivo;
  this->cromosoma=new size_t[this->tamanno];
  while(idx<this->tamanno){ *(this->cromosoma+idx) = *(ente.cromosoma+idx); ++idx; }
}

Individuo::~Individuo(){
  if(this->cromosoma) {delete[] this->cromosoma; this->cromosoma = nullptr;}
  this->tamanno = this->aptitud = this->coste = 0;
  this->vivo=false;
}

std::ostream& operator<< (std::ostream& os, const Individuo& ente){
  size_t idx;
  os << "[ ";
  idx=0; while(idx<ente.tamanno) os << *(ente.cromosoma+(idx++)) << "  ";
  os << "]" << std::endl
     << "Peso: " << ente.coste << std::endl
     << "Aptitud: " << ente.aptitud
  ;
  return os;
}

// Lovdog Población
Geneticos::Geneticos(){
  this->numGeneraciones=0;
  this->cantidadIndividuos=0;
  this->tipoCruce=0;
  this->mejorIndividuo=0;
  this->posiblesval=nullptr;
  this->poblacion=nullptr;
  this->nodos=nullptr;
  this->generador=nullptr;
  this->cruce=nullptr;
}

Geneticos::Geneticos(Geneticos& poblacion){
  this->numGeneraciones=poblacion.numGeneraciones;
  this->cantidadIndividuos=poblacion.cantidadIndividuos;
  this->tipoCruce=poblacion.tipoCruce;
  this->mejorIndividuo=poblacion.mejorIndividuo;
  this->posiblesval=nullptr;
  this->generador=poblacion.generador;
  this->cruce=poblacion.cruce;
  if(poblacion.nodos && poblacion.poblacion) {
    this->poblacion=new Individuo[this->cantidadIndividuos];
    *this->poblacion=*poblacion.poblacion;
    this->nodos=new lovdogListaNodos;
  }
}

Geneticos::Geneticos(const char* archivo){
  this->nodos = new lovdogListaNodos(archivo);
  this->poblacion=new Individuo[this->nodos->cardinalidad()];
  this->posiblesval=nullptr;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->mejorIndividuo=0;
  this->numGeneraciones=0;
  this->tipoCruce=0;
}

Geneticos::Geneticos(const lovdogListaNodos* listaNodos){
  this->nodos = new lovdogListaNodos(*listaNodos);
  this->poblacion=new Individuo[this->nodos->cardinalidad()];
  this->posiblesval=nullptr;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->mejorIndividuo=0;
  this->numGeneraciones=0;
  this->tipoCruce=0;
}

Geneticos::~Geneticos(){
  this->numGeneraciones=0;
  this->cantidadIndividuos=0;
  this->tipoCruce=0;
  this->mejorIndividuo=0;
  this->cruce=nullptr;
  this->generador=nullptr;
  if(this->posiblesval) {delete[] this->posiblesval; this->posiblesval=nullptr;}
  if(this->poblacion) {delete[] this->poblacion; this->poblacion=nullptr;}
  if(this->nodos) {delete this->nodos; this->nodos=nullptr;}
}

void Geneticos::estableceEvaluador(float (*evalFn)(size_t*,Grafo*)){ this->aptitudEvalFn=evalFn; }

void Geneticos::evaluar(void){
  size_t idx; idx=0;
  while(idx<this->cantidadIndividuos){
    (this->poblacion+idx)->aptitud = this->aptitudEvalFn((this->poblacion+idx)->cromosoma,this->nodos);
    ++idx;
  }
}

void Geneticos::define(const char campo, const void* valor){
  switch(campo){
    case AG_NUM_GENERACIONES:
      this->numGeneraciones=*((size_t*)valor);
      break;
    case AG_CANTIDAD_INDIVIDUOS:
      this->cantidadIndividuos=*((size_t*)valor);
      break;
    case AG_TIPO_CRUCE:
      this->tipoCruce=*((uint_t*)valor);
      break;
    case AG_CANTIDAD_CRUCES:
      this->crucesPorPoblacion=(size_t)valor;
      break;
  }
}

void Geneticos::cambiaTipoCruce(){
  switch(this->tipoCruce){
    case OX_CROSS:
      cruce = this->cruceOx;
      break;
    case PERMLANG_CROSS:
      cruce = this->crucePermLang;
      break;
  }
}

void Geneticos::cambiaTipoGenerador(void){
  switch (this->tipoIndividuo) {
    case INDIVIDUO_BINARIO:
      this->generador = generadorBinario;
      break;
    case INDIVIDUO_PERMUTACION:
      this->generador = generadorPermutado;
      break;
  }
}

void Geneticos::generadorPermutado(Individuo* poblacion, size_t cantidadIndividuos){
  // Genera individuos
  size_t idx,idy; idy=idx=0; while(idx<cantidadIndividuos){
    while (idy<(poblacion+idx)->tamanno)
      { *((poblacion+idx)->cromosoma+idy)=idy; idy++; }
    // Semilla aleatoria
    // Aleatorizando posiciones
    std::random_shuffle(
         (poblacion+idx)->cromosoma+idy, // Primer elemento
        ((poblacion+idx)->cromosoma+idy)+(poblacion+idx)->tamanno // Último elemento
    );
    ++idx;
  }
}

void Geneticos::generadorBinario(Individuo* poblacion, size_t cantidadIndividuos){
  return;
}

void Geneticos::cruceOx(const Individuo& padre, const Individuo& madre, Individuo* hijos){
  return;
}

void Geneticos::crucePermLang(const Individuo& padre, const Individuo& madre, Individuo* hijos){
  return;
}

std::ostream& operator<< (std::ostream& os, const Geneticos& g){
  size_t idx;
  os << "Cantidad de individuos por generación: " << std::endl
     << "Cantidad de generaciones: " << std::endl
     << "Número de generación: " << std::endl
     << "Número de cruces por generación: " << std::endl
     << "Mejor individuo de la generación: " << std::endl
     << "===Otros individuos===" << std::endl
  ;
  idx=0;while(idx < g.mejorIndividuo){ os << idx << ". " << *(g.poblacion+idx) << std::endl;
  } while((++idx) < g.mejorIndividuo)  os << idx << ". " << *(g.poblacion+idx) << std::endl;
  return os;
}

// Lovdog Lista Nodos
lovdogListaNodos::lovdogListaNodos(){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=0;
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=false;
  this->leeCSV(archivoCSV, CSV_SOLO_DATOS);
  this->creaMatrizAdyacencias();
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV, const char tipo){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=false;
  this->leeCSV(archivoCSV, tipo);
  this->creaMatrizAdyacencias();
}

lovdogListaNodos::lovdogListaNodos(lovdogListaNodos& lln){
  if(!(lln.dimensiones & lln.tamanno)) return;
  size_t idx,total;
  this->dimensiones = lln.dimensiones;
  this->tamanno     = lln.tamanno;
  this->tagged      = lln.tagged;
  this->x           = new float[this->dimensiones * this->tamanno];
  this->Adyacencias = new float[this->tamanno*this->tamanno];
  idx = 0; total = this->tamanno * this->dimensiones;
  while(idx < total) { *(this->x+idx) = *(lln.x+idx); ++idx; };
  idx = 0; total = this->tamanno * this->tamanno;
  while(idx < total) { *(this->Adyacencias+idx) = *(lln.Adyacencias+idx); ++idx; }
}

lovdogListaNodos::lovdogListaNodos(const lovdogListaNodos& lln){
  if(!(lln.dimensiones & lln.tamanno)) return;
  size_t idx,total;
  this->dimensiones = lln.dimensiones;
  this->tamanno     = lln.tamanno;
  this->tagged      = lln.tagged;
  this->x           = new float[this->dimensiones * this->tamanno];
  this->Adyacencias = new float[this->tamanno*this->tamanno];
  idx = 0; total = this->tamanno * this->dimensiones;
  while(idx < total) { *(this->x+idx) = *(lln.x+idx); ++idx; };
  idx = 0; total = this->tamanno * this->tamanno;
  while(idx < total) { *(this->Adyacencias+idx) = *(lln.Adyacencias+idx); ++idx; }
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
  while(auxIdy < *n){
    auxIdx=0;
    while(auxIdx < *n){
      *(matriz+auxIdx+(*n*auxIdy)) = this->distanciaEuclideana(auxIdx, auxIdy);
      ++auxIdx;
    }
    ++auxIdy;
  }

  if(this->Adyacencias != nullptr) delete[] this->Adyacencias;

  this->Adyacencias = matriz; matriz = nullptr;

  return true;
}

float lovdogListaNodos::distanciaEuclideana(size_t indexA, size_t indexB){
  if( indexA > this->tamanno ) indexA = indexA % this->tamanno;
  if( indexB > this->tamanno ) indexB = indexB % this->tamanno;
  size_t idx, total; float sum, *vectorX;

  sum=idx=0;
  total    = this->dimensiones;
  indexA  *= this->dimensiones;
  indexB  *= this->dimensiones;
  vectorX  = this->x;
  
  while(idx < total) {
    sum += powf( *(vectorX+indexB+idx) - *(vectorX+indexA+idx) , 2);
    ++idx;
  }

  return sqrtf(sum);
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
  size_t idx, xidx, moduloBase;

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
  
  switch(headers_indexes & CSV_CABECERA){
    case CSV_CABECERA:
      --this->tamanno;
      while(std::getline(token, linea, ','))
        ++this->dimensiones;
      break;
    case CSV_SOLO_DATOS:
      while(std::getline(token, linea, ','))
        if(!linea.empty() )
          valores.push_back(atof(linea.c_str()));
      this->dimensiones=valores.size();
      break;
  }
  token.str("");
  token.clear();

  switch(headers_indexes & CSV_INDICES){
    case CSV_INDICES:
      --this->dimensiones;
      inicializaX();
      if(!valores.empty()) {x[0]=valores[1]; x[1]=valores[2];}
      valores.clear();
      idx=xidx=0; moduloBase = this->dimensiones+1;
      while(std::getline(csvEnCuestion,linea)){ 
        token.str(""); token.clear(); token << linea ;
        while(std::getline(token,linea,',')) if((idx++)%moduloBase) *(this->x+(xidx++))=atof(linea.c_str());
      }
      break;
    case CSV_SOLO_DATOS:
      inicializaX();
      if(!valores.empty()) {x[0]=valores[0]; x[1]=valores[1];}
      xidx=0;
      while(std::getline(csvEnCuestion,linea)){
        token.str(""); token.clear(); token << linea ;
        while(std::getline(token,linea,',')) *(this->x+(xidx++))=atof(linea.c_str());
      }
      break;
  }
  csvEnCuestion.close();
}

void lovdogListaNodos::inicializaX(void){
  size_t idx, total;
  idx = 0;
  total = (this->dimensiones)*(this->tamanno);
  this->x = new float[total];
  if(this->x == nullptr){  this->tamanno = this->dimensiones = 0; return; }
  while(idx < total) *(this->x+(idx++)) = 0;
}

float* lovdogListaNodos::nodoEn(size_t idx)      const { return  (this->x+(idx*this->dimensiones)); }
float* lovdogListaNodos::operator ()(size_t idx) const { return  (this->x+(idx*this->dimensiones)); }
float  lovdogListaNodos::Xa(size_t idx)          const { return *(this->x+idx); }
float  lovdogListaNodos::operator [](size_t idx) const { return *(this->x+idx); }
size_t lovdogListaNodos::dimensionPorNodo(void)  const { return dimensiones; }
size_t lovdogListaNodos::cardinalidad(void)      const { return tamanno; }

std::ostream& operator <<(std::ostream& os, const lovdogListaNodos& grafo){
  size_t xidx,total;
  xidx=0;
  total=grafo.cardinalidad()*grafo.dimensionPorNodo();
  os << "[";
  while(xidx < total) (xidx)%grafo.dimensionPorNodo() ? os << grafo[xidx++] << "\t" : os << "\n" << grafo[xidx++] << "\t";
  os << "\n]";
  return os;
}

bool lovdogListaNodos::imprimeMatrizAdyacencias(void){
  if(this->Adyacencias == nullptr) return false;
  size_t index,dims,total;
  index = 0;
  total = this->tamanno * this->tamanno;
  dims  = this->tamanno;
  while(index < total)
    index%dims? std::cout << *(this->Adyacencias+(index++)) << "\t" : std::cout << std::endl << *(this->Adyacencias+(index++)) << "\t";
  return true;
}

} // Namespace End
