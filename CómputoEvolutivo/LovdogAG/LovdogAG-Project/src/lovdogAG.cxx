#include "lovdogAG.hxx"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <ctime>
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
  this->vivo=false;
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

Individuo::Individuo(size_t n, bool numerado=false){
  size_t idx;
  this->aptitud=0;
  this->coste=0;
  this->tamanno=n;
  this->vivo=false;
  this->cromosoma= new size_t[n];
  idx=0; if(numerado) while(idx<n){ *(this->cromosoma+idx) = idx; ++idx;}
         else while(idx<n) *(this->cromosoma+(idx++)) = 0;
}

Individuo::Individuo(Individuo& ente){
  if(!(ente.tamanno && ente.cromosoma)) return;
  size_t idx;
  this->tamanno=ente.tamanno;
  this->aptitud=ente.aptitud;
  this->coste=ente.coste;
  this->vivo=ente.vivo;
  this->cromosoma=new size_t[this->tamanno];
  idx=0; while(idx<this->tamanno){ *(this->cromosoma+idx) = *(ente.cromosoma+idx); ++idx; }
}

Individuo::~Individuo(){
  if(this->cromosoma) {delete[] this->cromosoma; this->cromosoma = nullptr;}
  this->tamanno = this->aptitud = this->coste = 0;
  this->vivo=false;
}

bool Individuo::existe(const Individuo* ente, size_t* nucleotido){
  size_t idx;
  idx=0; while(idx < ente->tamanno) if(*(ente->cromosoma+(idx++)) == *nucleotido) return true;
  return false;
}

Individuo& Individuo::operator=(const Individuo& ente) {
  if(!(ente.tamanno && ente.cromosoma)) return *this;
  if(this->cromosoma) delete []  this->cromosoma;
  size_t idx;
  this->cromosoma=nullptr;
  this->tamanno=ente.tamanno;
  this->aptitud=ente.aptitud;
  this->coste=ente.coste;
  this->vivo=ente.vivo;
  this->cromosoma=new size_t[this->tamanno];
  idx=0; while(idx<this->tamanno){ *(this->cromosoma+idx) = *(ente.cromosoma+idx); ++idx; }
  return *this;
}

std::ostream& operator<< (std::ostream& os, const Individuo& ente){
  size_t idx;
  os << "[ ";
  idx=0; while(idx<ente.tamanno) os << *(ente.cromosoma+(idx++)) << "  ";
  os << "]" << std::endl
     << "Peso: "         LDP << ente.coste   << std::endl
     << "Aptitud: "      LDP << ente.aptitud << std::endl
     << "Cardinalidad: " LDP << ente.tamanno
  ;
  return os;
}

// Lovdog Lista Nodos
lovdogListaNodos::lovdogListaNodos(){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->m=0;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=0;
  this->soloAdyacencias=false;
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->m=0;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=false;
  this->leeCSV(archivoCSV, CSV_SOLO_DATOS);
  this->soloAdyacencias=false;
  this->creaMatrizAdyacencias();
}

lovdogListaNodos::lovdogListaNodos(const char* archivoCSV, const char tipo){
  this->Adyacencias=nullptr;
  this->x=nullptr;
  this->m=0;
  this->tamanno=0;
  this->dimensiones=0;
  this->tagged=false;
  this->soloAdyacencias=false;
  if(tipo & this->CSV_ADYACENCIAS_INPUT ){
    this->creaMatrizAdyacencias(archivoCSV,tipo);
    return;
  }
  this->leeCSV(archivoCSV, tipo);
  this->creaMatrizAdyacencias();
}

lovdogListaNodos::lovdogListaNodos(lovdogListaNodos& lln){
  if(!(lln.tamanno)) return;
  size_t idx,total;
  this->dimensiones     = lln.dimensiones;
  this->tamanno         = lln.tamanno;
  this->tagged          = lln.tagged;
  this->soloAdyacencias = lln.soloAdyacencias;
  this->x               = new float[this->dimensiones * this->tamanno];
  this->m=0;
  this->Adyacencias     = new float[this->tamanno*this->tamanno];
  idx = 0; total        = this->tamanno * this->dimensiones;
  while(idx < total) { *(this->x+idx) = *(lln.x+idx); ++idx; };
  idx = 0; total        = this->tamanno * this->tamanno;
  while(idx < total) { *(this->Adyacencias+idx) = *(lln.Adyacencias+idx); ++idx; }
}

lovdogListaNodos::lovdogListaNodos(const lovdogListaNodos& lln){
  if(!(lln.tamanno)) return;
  size_t idx,total;
  this->dimensiones     = lln.dimensiones;
  this->tamanno         = lln.tamanno;
  this->tagged          = lln.tagged;
  this->soloAdyacencias = lln.soloAdyacencias;
  this->x               = new float[this->dimensiones * this->tamanno];
  this->m=0;
  this->Adyacencias     = new float[this->tamanno*this->tamanno];
  idx = 0; total        = this->tamanno * this->dimensiones;
  while(idx < total) { *(this->x+idx) = *(lln.x+idx); ++idx; };
  idx = 0; total        = this->tamanno * this->tamanno;
  while(idx < total) { *(this->Adyacencias+idx) = *(lln.Adyacencias+idx); ++idx; }
}

lovdogListaNodos::~lovdogListaNodos(){
  if(this->x) delete[] this->x;
  if(this->Adyacencias) delete[] this->Adyacencias;
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
  if(!csvEnCuestion.is_open()){ std::cerr << "Error al leer CSV"; return; }
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

bool lovdogListaNodos::creaMatrizAdyacencias(std::string nombreArchivo, const char headers_indexes){
  std::fstream csvEnCuestion;
  std::stringstream token;
  std::string linea;
  std::vector<float> valores;
  size_t idx, xidx, moduloBase,total;
  this->soloAdyacencias=true;

  csvEnCuestion.open(nombreArchivo, std::ios::in);
  if(!csvEnCuestion.is_open()){ std::cerr << "Error al leer CSV"; return false; }
  this->tamanno = std::count(
      std::istreambuf_iterator<char>(csvEnCuestion),
      std::istreambuf_iterator<char>(), '\n'
  );
  csvEnCuestion.clear();
  csvEnCuestion.seekg(0);

  // En esta versión, this->tamanno es la única variable utilizable

  std::getline(csvEnCuestion, linea);
  token = std::stringstream(linea);
  
  switch(headers_indexes & CSV_CABECERA){
    case CSV_CABECERA:
      --this->tamanno;
      while(std::getline(token, linea, ','))
        if(!linea.empty() ) valores.push_back(0); /*Keep count of every column*/
      if(valores.size() < this->tamanno) this->tamanno = valores.size();
      /*If there are inconsistent dims, go for the smallest one*/
      valores.clear(); std::vector<float>().swap(valores);
      /*Full cleaning the vector, this might be too much computational cost
       * but needed being a control variable in the next section*/
      break;
    case CSV_SOLO_DATOS:
      while(std::getline(token, linea, ','))
        if(!linea.empty() )
          valores.push_back(atof(linea.c_str()));
      if(valores.size() < this->tamanno) this->tamanno = valores.size();
      break;
  }
  token.str("");
  token.clear();
  /*Those two above are for cleaning the token container and restarting its iterator
   * yeah, too much, not that clear what or why. Could do it more understandable?*/

  /* Well, we are here now, there's a part where everything seems like some noodle
   * without a little bit of common sense, in my defense, it was my first attempt
   * maybe the methods should be more efficient.*/
  /* --> For context, here is the part where the data is stored, but it's 
   * mixed with the indexes headers control flow, I don't want any useless stacking
   * however, maybe I'll reconsider it for the sake of clearness and less redundant code */
  switch(headers_indexes & CSV_INDICES){
    case CSV_INDICES: // This part is when the file is known to have an index column
      if(!(this->CSV_CABECERA & headers_indexes)) --this->tamanno;
      inicializaM(); // Create matrix, because the size is clearly good calculated by now
      if(!valores.empty()) { /* Remember the clearing of this variable, here's the why */
        idx=0; while(idx < this->tamanno){
          *(this->Adyacencias+idx)=valores[idx]; /*Storing data gotten in the last section*/
          ++idx;
        }
      }
      valores.clear();
      idx=xidx=0; moduloBase = this->tamanno+1; total = this->tamanno*this->tamanno;
      while(std::getline(csvEnCuestion,linea)){ 
        token.str(""); token.clear(); token << linea; /*Limpieza del token*/
        while(std::getline(token,linea,',')&&xidx<total){
          if((idx++)%moduloBase) *(this->Adyacencias+(xidx++))=atof(linea.c_str()); /*Asignación de valores de matriz*/
      }}
      break; // At this point, here's the ending
    case CSV_SOLO_DATOS: // This part is when the file is known to not to have an index column
      inicializaM();// Déjà vu !, est-ce que vous ne pensez pas comme ça ?
      if(!valores.empty()) { /* Remember the clearing of this variable, here's the why */
        idx=0; while(idx < this->tamanno){
          *(this->Adyacencias+idx)=valores[idx]; /*Storing data gotten in the last section*/
          ++idx;
        }
      }
      valores.clear();
      xidx=0;
      while(std::getline(csvEnCuestion,linea)){
        token.str(""); token.clear(); token << linea; /*Limpieza del token*/
        while(std::getline(token,linea,',')&&xidx<this->tamanno)
          *(this->Adyacencias+(xidx++))=atof(linea.c_str()); // Here's the storing, do you remember?
      }
      break; // Well, here's the alternative ending, tho
  }
  csvEnCuestion.close();
  this->penalizacionMatriz();
  return true;
}

void lovdogListaNodos::inicializaX(void){
  size_t idx, total;
  idx = 0;
  total = (this->dimensiones)*(this->tamanno);
  this->x = new float[total];
  if(this->x == nullptr){  this->tamanno = this->dimensiones = 0; return; }
  while(idx < total) *(this->x+(idx++)) = 0;
}

void lovdogListaNodos::inicializaM(void){
  size_t idx, total;
  idx = 0;
  total = (this->tamanno)*(this->tamanno);
  this->Adyacencias = new float[total];
  if(this->Adyacencias == nullptr){  this->tamanno = this->dimensiones = 0; return; }
  while(idx < total) *(this->Adyacencias+(idx++)) = 0;
}

void lovdogListaNodos::penalizacionMatriz(void){
  // Here now, for each one, the penalization will be the 50% of the sum of all values
  size_t idx,total;
  float sum;
  total = this->tamanno*this->tamanno;
  this->m=
  sum=(float)(idx=0);
  while(idx<total) sum += (int)*(this->Adyacencias+(idx++)) << 1;
  idx=0;
  while(idx<total){ if(!*(this->Adyacencias+idx)) *(this->Adyacencias+idx)=sum; ++idx; };
}

float* lovdogListaNodos::nodoEn(size_t idx)      const { return  (this->x+(idx*this->dimensiones)); }
float* lovdogListaNodos::operator ()(size_t idx) const { return  (this->x+(idx*this->dimensiones)); }
float  lovdogListaNodos::Xa(size_t idx)          const { return *(this->x+idx); }
float  lovdogListaNodos::MaxPossible(void)       const { return   this->m;}
float  lovdogListaNodos::operator [](size_t idx) const { return *(this->x+idx); }
size_t lovdogListaNodos::dimensionPorNodo(void)  const { return dimensiones; }
size_t lovdogListaNodos::cardinalidad(void)      const { return tamanno; }

std::ostream& operator <<(std::ostream& os, const lovdogListaNodos& grafo){
  if(grafo.soloAdyacencias){
    os << " [  ] \n";
    grafo.imprimeMatrizAdyacencias();
    return os;
  };
  size_t xidx,total;
  xidx=0;
  total=grafo.cardinalidad()*grafo.dimensionPorNodo();
  os << "[";
  while(xidx < total) (xidx)%grafo.dimensionPorNodo() ? os LDP << grafo[xidx++] << "\t" : os << "\n" LDP << grafo[xidx++] << "\t|\t";
  os << "\n]";
  return os;
}

bool lovdogListaNodos::imprimeMatrizAdyacencias(void) const{
  if(this->Adyacencias == nullptr) return false;
  size_t index,dims,total;
  index = 0;
  total = this->tamanno * this->tamanno;
  dims  = this->tamanno;
  while(index < total)
    index%dims? std::cout << *(this->Adyacencias+(index++)) << "\t" : std::cout << std::endl << *(this->Adyacencias+(index++)) << "\t|\t";
  return true;
}

lovdogListaNodos& lovdogListaNodos::operator=(const lovdogListaNodos& lln){
  if(!(this->tamanno)) return *this;
  size_t idx,total;
  this->dimensiones     = lln.dimensiones;
  this->tamanno         = lln.tamanno;
  this->tagged          = lln.tagged;
  this->soloAdyacencias = lln.soloAdyacencias;
  this->x               = new float[this->dimensiones * this->tamanno];
  this->Adyacencias     = new float[this->tamanno*this->tamanno];
  idx = 0; total        = this->tamanno * this->dimensiones;
  while(idx < total) { *(this->x+idx) = *(lln.x+idx); ++idx; };
  idx = 0; total        = this->tamanno * this->tamanno;
  while(idx < total) { *(this->Adyacencias+idx) = *(lln.Adyacencias+idx); ++idx; }
  return *this;
}

// Lovdog Población
Geneticos::Geneticos(){
  this->numGeneraciones=0;
  this->cantidadIndividuos=0;
  this->tipoCruce=0;
  this->tipoSeleccion=0;
  this->mejorIndividuo=0;
  this->mejorRendimiento=0;
  this->generacionActual=0;
  this->verbose=0;
  this->mejorRendimientoHistorico=0;
  this->mejorRendimientoHistoricoUsado=false;;
  this->aptitudEvalFn=nullptr;
  this->nPoblacion = nullptr;
  this->posiblesval=nullptr;
  this->poblacion=nullptr;
  this->nodos=nullptr;
  this->generador=nullptr;
  this->cruce=nullptr;
  std::srand(std::time(0));
}

Geneticos::Geneticos(Geneticos& poblacion){
  this->numGeneraciones=poblacion.numGeneraciones;
  this->cantidadIndividuos=poblacion.cantidadIndividuos;
  this->tipoCruce=poblacion.tipoCruce;
  this->tipoSeleccion=poblacion.tipoSeleccion;
  this->mejorIndividuo=poblacion.mejorIndividuo;
  this->posiblesval=nullptr;
  this->mejorRendimientoHistorico=poblacion.mejorRendimientoHistorico;
  this->mejorRendimientoHistoricoUsado=poblacion.mejorRendimientoHistoricoUsado;
  this->aptitudEvalFn=poblacion.aptitudEvalFn;
  this->generacionActual=poblacion.generacionActual;
  this->verbose=poblacion.verbose;
  this->mejorhst=poblacion.mejorhst;
  this->generador=poblacion.generador;
  this->mejorRendimiento=poblacion.mejorRendimiento;
  this->cruce=poblacion.cruce;
  if(poblacion.nodos && poblacion.poblacion) {
    this->poblacion=new Individuo[this->cantidadIndividuos];
    *this->poblacion=*poblacion.poblacion;
    this->nodos=new lovdogListaNodos;
  }
  this->nPoblacion=nullptr;
  std::srand(std::time(0));
}

Geneticos::Geneticos(const char* archivo){
  this->nodos = new lovdogListaNodos(archivo);
  this->poblacion=nullptr;
  this->posiblesval=nullptr;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->aptitudEvalFn=nullptr;
  this->nPoblacion = nullptr;
  this->mejorRendimientoHistorico=0;
  this->mejorRendimientoHistoricoUsado=false;
  this->mejorIndividuo=0;
  this->generacionActual=0;
  this->verbose=0;
  this->numGeneraciones=0;
  this->tipoCruce=0;
  this->tipoSeleccion=0;
  this->mejorRendimiento=0;
  std::srand(std::time(0));
}

Geneticos::Geneticos(const char* archivo, const char tipo){
  this->nodos = new lovdogListaNodos(archivo,tipo);
  this->poblacion=nullptr;
  this->posiblesval=nullptr;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->aptitudEvalFn=nullptr;
  this->nPoblacion = nullptr;
  this->mejorRendimientoHistorico=0;
  this->mejorRendimientoHistoricoUsado=false;
  this->mejorIndividuo=0;
  this->generacionActual=0;
  this->verbose=0;
  this->numGeneraciones=0;
  this->tipoCruce=0;
  this->tipoSeleccion=0;
  this->mejorRendimiento=0;
  std::srand(std::time(0));
}

Geneticos::Geneticos(const lovdogListaNodos* listaNodos){
  this->nodos = new lovdogListaNodos(*listaNodos);
  this->poblacion=new Individuo[this->nodos->cardinalidad()];
  this->posiblesval=nullptr;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->aptitudEvalFn=nullptr;
  this->nPoblacion = nullptr;
  this->mejorRendimientoHistorico=0;
  this->mejorRendimientoHistoricoUsado=false;
  this->mejorIndividuo=0;
  this->generacionActual=0;
  this->verbose=0;
  this->numGeneraciones=0;
  this->tipoCruce=0;
  this->tipoSeleccion=0;
  this->mejorRendimiento=0;
  std::srand(std::time(0));
}

Geneticos::~Geneticos(){
  this->numGeneraciones=0;
  this->cantidadIndividuos=0;
  this->tipoCruce=0;
  this->mejorRendimientoHistorico=0;
  this->mejorRendimientoHistoricoUsado=false;
  this->generacionActual=0;
  this->verbose=0;
  this->tipoSeleccion=0;
  this->mejorIndividuo=0;
  this->cruce=nullptr;
  this->generador=nullptr;
  this->aptitudEvalFn=nullptr;
  this->nPoblacion = nullptr;
  if(this->posiblesval) {delete[] this->posiblesval; this->posiblesval=nullptr;}
  if(this->poblacion) {delete[] this->poblacion; this->poblacion=nullptr;}
  if(this->nPoblacion) {delete[] this->nPoblacion; this->nPoblacion=nullptr;}
  if(this->nodos) {delete this->nodos; this->nodos=nullptr;}
}


void Geneticos::ejecuta (uchar_t evento){
  switch(evento){
    case AG_PRIMERA_GENERACION:
      //std::cout << "Primera Generación"<<std::endl;
      if(!generador) {std::cerr << "Sin generador definido"; return;}
      generador(&this->poblacion,this->cantidadIndividuos, this->nodos);
      //std::cout << "Primera Generación"<<std::endl;
      break;
    case AG_CRUCE:
      //std::cout << "Cruce"<<std::endl;
      if(!this->poblacion && !this->crucesPorPoblacion) { std::cerr << "Sin población o cantidad de cruces definido"; return; }
      if(!this->cruce){ std::cerr << "Sin función de cruce definida"; return; }
      temporadaApareamiento();
      //std::cout << "Cruce"<<std::endl;
      break;
    case AG_DEPREDADOR:
      //std::cout << "Depredador"<<std::endl;
      if(!this->poblacion && !this->crucesPorPoblacion) { std::cerr << "Sin población o cantidad de cruces definido"; return; }
      depredadorMatadorAtak(poblacion);
      //std::cout << "Depredador"<<std::endl;
      break;
    case AG_SELECCION:
      break;
    case AG_REEMPLAZO:
      //std::cout << "Reemplazo de individuos"<<std::endl;
      if(!this->poblacion && !this->crucesPorPoblacion) { std::cerr << "Sin población o cantidad de cruces definido"; return; }
      if(!this->nPoblacion) { std::cerr << "Sin población nueva"; return; }
      reemplazo(poblacion,nPoblacion,cantidadIndividuos,crucesPorPoblacion<<1);
      //std::cout << "Reemplazo de individuos"<<std::endl;
      break;
    case AG_EVALUACION:
      //std::cout << "Evaluación de individuos"<<std::endl;
      if(!this->poblacion) { std::cerr << "Sin población definida"; return; }
      if(!this->nodos) { std::cerr << "Sin nodos definidos"; return; }
      evaluar();
      //std::cout << "Evaluación de individuos"<<std::endl;
      break;
    case AG_MUTACION:
      //std::cout << "Mutación de individuos"<<std::endl;
      if(!this->poblacion) { std::cerr << "Sin población definida"; return; }
      if(!this->nodos) { std::cerr << "Sin nodos definidos"; return; }
      mutacion(this->poblacion,this->mutacionesPorGeneracion);
      //std::cout << "Mutación de individuos"<<std::endl;
      break;
    case AG_INICIA:
      this->cicloSimple();
      break;
  }
}

void Geneticos::define(const char campo, uint_t valor){
  switch(campo){
    case AG_TIPO_CRUCE:
      this->tipoCruce=valor;
      this->cambiaTipoCruce();
      break;
    case AG_TIPO_SELECCION:
      this->tipoSeleccion=valor;
      this->cambiaTipoSeleccion();
      break;
    case AG_TIPO_INDIVIDUO:
      this->tipoIndividuo=valor;
      cambiaTipoGenerador();
      break;
    case AG_TIPO_REEMPLAZO:
      this->tipoReemplazo=valor;
      cambiaTipoReemplazo();
      break;
    case AG_VERBOSITY:
      this->verbose=valor;
      break;
  }
}
void Geneticos::define(const char campo, size_t valor){
  switch(campo){
    case AG_NUM_GENERACIONES:
      this->numGeneraciones=valor;
      break;
    case AG_CANTIDAD_INDIVIDUOS:
      this->cantidadIndividuos=valor;
      break;
    case AG_CANTIDAD_CRUCES:
      this->crucesPorPoblacion=
        valor>(this->cantidadIndividuos/2)?
        (this->cantidadIndividuos/2):valor;
      break;
    case AG_CANTIDAD_MUTACIONES:
      this->mutacionesPorGeneracion=(size_t)valor;
      break;
  }
}

void Geneticos::estableceEvaluador(float (*evalFn)(size_t*,Grafo*)){ this->aptitudEvalFn=evalFn; }


void Geneticos::evaluar(void){
  if(!this->aptitudEvalFn){ std::cerr << "Función de evaluación no definida" << std::endl; return; }
  if(!this->mejorRendimientoHistoricoUsado){
    this->mejorRendimientoHistorico=this->aptitudEvalFn((this->poblacion)->cromosoma,this->nodos);
    this->mejorRendimientoHistoricoUsado=true;
  }
  size_t idx; idx=0;
  this->mejorRendimiento = (this->poblacion+idx)->aptitud;
  while(idx<this->cantidadIndividuos){
    (this->poblacion+idx)->aptitud = this->aptitudEvalFn((this->poblacion+idx)->cromosoma,this->nodos);
    if((this->poblacion+idx)->aptitud < this->mejorRendimiento){
      this->mejorRendimiento = (this->poblacion+idx)->aptitud;
      this->mejorIndividuo=idx;
    }
    if((this->poblacion+idx)->aptitud < this->mejorRendimientoHistorico ) {
      this->mejorRendimientoHistorico = (this->poblacion+idx)->aptitud;
      this->mejorhst = *(this->poblacion+idx);
    }
    ++idx;
  }
}

void Geneticos::temporadaApareamiento(void){
  size_t  numeroHijos = this->crucesPorPoblacion << 1;
  size_t* seleccionados = new size_t[2];
  this->nPoblacion=new Individuo[numeroHijos];
  size_t opciones[4] = { 0,0,0,0 };
  size_t contador;

  contador=0; while(contador<numeroHijos)
    *(this->nPoblacion+(contador++)) = Individuo(this->poblacion->tamanno);
  opciones[0]= this->cantidadIndividuos; opciones[1]= 2;
  printf("\b/");
  while((contador++) < this->crucesPorPoblacion){
    printf("\b|");
    opciones[2]= (size_t)std::rand()%2;
    opciones[3]= (size_t)std::rand()%(this->cantidadIndividuos-2);
    printf("\b\\");
    this->seleccion(this->poblacion, seleccionados, opciones);
    printf("\b-");
    this->cruce(
      this->poblacion+*(seleccionados  ),
      this->poblacion+*(seleccionados+1),
      this->nPoblacion,
      numeroHijos
    );
    printf("\b/");
  }
}

void Geneticos::depredadorMatadorAtak(Individuo* poblacion){
  size_t idx; idx=0;
  reiniciadorDIndividuo(*(poblacion+idx));
  return;
}


/*void Geneticos::choqueGeneracionalAleatorio(Individuo* poblacion,Individuo* nPoblacion, size_t tamanno){
//  size_t contador, ndh, idx, idh; char *idb;
//  // idh se usará como contador para los índices de la nueva generación
//  // idx se usará como contador para los índices de la generación actual
//  // La actualización se hará siempre y cuando sea para mejorar la raza
//  idb=new char[this->cantidadIndividuos];
//  contador=0; while(contador < this->cantidadIndividuos ) *(idb+(contador++)) = 0;
//  idh=contador=0; ndh=this->crucesPorPoblacion<<1;
//  while( (contador++) < this->crucesPorPoblacion && idh < ndh ){
//    idx=(size_t)std::rand()%this->cantidadIndividuos;
//    if(*(idb+idx)) {std::cout << "."; continue; }
//    if((this->nPoblacion+idh)->vivo) // Si está vivo, se continúa con la asignación
//      if((this->nPoblacion+idh)->aptitud < (this->poblacion+idx)->aptitud)
//        *(this->poblacion+idx)=*(this->nPoblacion+idh);
//    ++idh; // Siguiente hijo
//    ++*(idb+idx); // Se cuenta como usado
//  } 
//  delete[] idb;
//  delete[] this->nPoblacion;
//  this->nPoblacion=nullptr;
}*/

void Geneticos::choqueGeneracionalElitista(Individuo* poblacion,Individuo* nPoblacion, size_t cantidadIndividuos, size_t cantidadHijos){
  // populacion es la cantidad de individuos
  size_t* Seleccionados; size_t i,np;
  size_t opciones[4] = {cantidadIndividuos, cantidadIndividuos, 0, 0};
  Individuo* poblacionMezclada;
  np=cantidadIndividuos+cantidadHijos;
  poblacionMezclada = new Individuo[np];
  Seleccionados = new size_t[cantidadIndividuos];
  i=0; while(i<cantidadIndividuos) {*(poblacionMezclada+i) = *(poblacion+i); ++i;}
  i=0; while(i<cantidadHijos)      {*(poblacionMezclada+i+cantidadIndividuos) = *(nPoblacion+i); ++i;}
  Geneticos::elitismo(poblacionMezclada, Seleccionados, opciones);
  i=0; while(i<cantidadIndividuos) { *(poblacion+i) = *(poblacionMezclada+i); ++i; }
  //i=0; while(i<cantidadIndividuos) { std::cout << *(poblacion+i); ++i; }
}


void Geneticos::reiniciadorDIndividuo(const Individuo* ente, size_t mutaciones){
  size_t contador,idx,idy;
  if(mutaciones > ente->tamanno) mutaciones = ente->tamanno;
  contador=0; while(contador < mutaciones ){
    idx=(std::rand()+std::rand())%ente->tamanno;
    idy=(std::rand()+std::rand())%ente->tamanno;
    if(idx==idy) continue;
    *(ente->cromosoma+idx)^=
    *(ente->cromosoma+idy);
    *(ente->cromosoma+idy)^=
    *(ente->cromosoma+idx);
    *(ente->cromosoma+idx)^=
    *(ente->cromosoma+idy);
    ++contador;
  }
}

void Geneticos::reiniciadorDIndividuo(Individuo& ente){
  size_t contador,idx,idy;
  contador=0;
  while(contador < ente.tamanno ){
    idx=(std::rand()+std::rand())%ente.tamanno;
    idy=(std::rand()+std::rand())%ente.tamanno;
    if(idx==idy) continue;
    *(ente.cromosoma+idx)^=
    *(ente.cromosoma+idy);
    *(ente.cromosoma+idy)^=
    *(ente.cromosoma+idx);
    *(ente.cromosoma+idx)^=
    *(ente.cromosoma+idy);
    ++contador;
  }
}

void Geneticos::elitismo(const Individuo* poblacion, size_t* seleccionados, size_t* opciones){
  // Las opciones de esta función son
  //// [0]=> cantidad de individuos en la población;
  //// [1]=> cantidad de individuos a seleccionar;
  //// [2]=> 0/1 ascendente/descendente;
  //// [3]=> punto partida
  if(!*(opciones+1) || !*(opciones+0)) return;
  size_t* auxiliar;
  size_t idx,idy,limitx;
  if(*(opciones+1)+*(opciones+3) > *(opciones+0))
    *(opciones+3) = *(opciones+0) - *(opciones+1);
  auxiliar = new size_t[*(opciones)];
  idy=0; while(idy<*(opciones)) { *(auxiliar+idy)=idy; ++idy; }
  // Ordenamiento de elementos (índices)
  idx=0; limitx=*(opciones)-1;
  while(idx<limitx){
    idy=idx+1;
    while(idy<*(opciones)){
      if((poblacion+*(auxiliar+idx))->aptitud < (poblacion+*(auxiliar+idy))->aptitud ){
        *(auxiliar+idx)^=*(auxiliar+idy);
        *(auxiliar+idy)^=*(auxiliar+idx);
        *(auxiliar+idx)^=*(auxiliar+idy);
      } ++idy;
    } ++idx;
  }
  if(!seleccionados){
    std::cerr << "Array \"seleccionados\" sin inicialización" << std::endl;
    delete[] auxiliar;
    return;
  }
  if(*(opciones+2)==0){
    idx=0; while(idx<*(opciones+1)){
      *(seleccionados+idx) = *(auxiliar+idx+*(opciones+3));
      ++idx;
    }
    delete [] auxiliar;
    return;
  }
  limitx+=*(opciones+3);
  idx=0; while(idx<*(opciones+1)){
    *(seleccionados+idx) = *(auxiliar+limitx-idx);
    ++idx;
  }
  delete [] auxiliar;
  return;
}


void Geneticos::cambiaTipoCruce(void){
  switch(this->tipoCruce){
    case CRUCE_OX_CROSS:
      cruce = this->cruceOx;
      break;
    case CRUCE_PERMLANG_CROSS:
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
      this->mutacion = reiniciadorDIndividuo;
      break;
  }
}

void Geneticos::cambiaTipoSeleccion(void){
  switch(this->tipoSeleccion){
    case SELECCION_ELITISMO:
      seleccion=elitismo;
      break;
  }
}

void Geneticos::cambiaTipoReemplazo(void){
  switch(this->tipoReemplazo){
    case REEMPLAZO_ELITISMO:
      reemplazo=choqueGeneracionalElitista;
      break;
    //case REEMPLAZO_ALEATORIO:
    //  reemplazo=choqueGeneracionalAleatorio;
    //  break;
  }
}


void Geneticos::generadorPermutado(Individuo** Poblacion, size_t cantidadIndividuos, const Grafo* nodos){
  if(*Poblacion) { std::cerr << "Población ya creada" << std::endl; return; }
  size_t idx;
  // Reserva memoria
  *Poblacion = new Individuo[cantidadIndividuos];
  // Genera individuos
  idx=0; while(idx<cantidadIndividuos){
    // Inicialización del individuo
    *(*Poblacion+idx) = Individuo(nodos->cardinalidad(),true);
    // Aleatorizando elementos del individuo
    reiniciadorDIndividuo(*(*Poblacion+idx));
    ++idx;
  }
}

void Geneticos::generadorBinario(Individuo** poblacion, size_t cantidadIndividuos, const Grafo* nodos){
  return;
}


void Geneticos::cruceOx(const Individuo* padre, const Individuo* madre, Individuo* hijos, size_t ndh){
  size_t puntoCruce, idx,idh,idi,idj;
  puntoCruce = 3; //std::rand()%padre.tamanno;
  idx=idh=idj=0; idi=puntoCruce;
  while(idh<ndh && (hijos+idh)->vivo ) ++idh; //Búsqueda del individuo aún no inicializado
  if(idh > ndh-2) {std::cerr << "Imposible realizar cruza" << std::endl; return;}
  if(!(ndh-2 > idh)) return;
  while(idx<puntoCruce)     { *((hijos+idh  )->cromosoma+idx) = *(padre->cromosoma+idx); ++idx; }
  while(idx<padre->tamanno) { *((hijos+idh+1)->cromosoma+idx) = *(padre->cromosoma+idx); ++idx; }
  while((idx--) > 0){
    if(Individuo::existe(hijos+idh, madre->cromosoma+idx)){
      *((hijos+idh+1)->cromosoma+(idj++)) = *(madre->cromosoma+idx); continue;
    } *((hijos+idh  )->cromosoma+(idi++)) = *(madre->cromosoma+idx);
  }
  (hijos+idh+1)->vivo=true;
  (hijos+idh  )->vivo=true;
  return;
}

void Geneticos::crucePermLang(const Individuo* padre, const Individuo* madre, Individuo* hijos, size_t ndh){
  return;
}


void Geneticos::cicloSimple(void){
  std::cout << "Iniciando proceso genérico\n";
  this->generacionActual=0;
  this->ejecuta(this->AG_PRIMERA_GENERACION);
  if(this->verbose) while((this->generacionActual++)<this->numGeneraciones){
    this->ejecuta(AG_EVALUACION);
    this->ejecuta(AG_CRUCE);
    this->ejecuta(AG_REEMPLAZO);
    this->ejecuta(AG_MUTACION);
    std::cout << *this;
  }
  else while((this->generacionActual++)<this->numGeneraciones){
    std::cout << "Iteración " << this->generacionActual << std::endl;
    this->ejecuta(AG_EVALUACION);
    this->ejecuta(AG_CRUCE);
    this->ejecuta(AG_REEMPLAZO);
  }
  std::cout << "Fin del proceso"<<std::endl;
}

float Geneticos::TSPEvaluador(size_t* camino,Grafo* nodos){
  float sum; size_t idx;
  sum=0; idx=nodos->cardinalidad()-1;
  while(idx--) sum+=*nodos->celdaMatrizAdyacencias(*(camino+idx),*(camino+idx+1));
  return sum;
}

lovdogListaNodos Geneticos::grafo(void) const{
  lovdogListaNodos salida = lovdogListaNodos(*(this->nodos));
  return salida;
}

Individuo Geneticos::elIndividuo(size_t indice){
  if(indice>this->cantidadIndividuos-1) indice%=this->cantidadIndividuos;
  return *(this->poblacion+indice);
}


std::ostream& operator<< (std::ostream& os, const Geneticos& g){
  size_t idx;
  os
     << "<Descripción de Población>"                                          << std::endl
     << "Número de generación: "                  LDP << g.generacionActual-1 << std::endl
     << "Cantidad de individuos por generación: " LDP << g.cantidadIndividuos << std::endl
     << "Cantidad de generaciones: "              LDP << g.numGeneraciones    << std::endl
     << "Número de cruces por generación: "       LDP << g.crucesPorPoblacion << std::endl
  ;
  if(g.poblacion){
    os << "Mejor rendimiento histórico: "      LDP << g.mejorhst << std::endl
       << "Mejor individuo de la generación: " LDP << g.mejorIndividuo << ". "  // ↴
                                               LDP << *(g.poblacion+g.mejorIndividuo) << std::endl
    ;
    if(g.verbose == g.VERBOSITY_GENERACIONES){
      os << "===Otros individuos===" << std::endl;
      idx=0;while(idx < g.mejorIndividuo    ){ os << idx << ". " << *(g.poblacion+idx) << std::endl;
      } while((++idx) < g.cantidadIndividuos)  os << idx << ". " << *(g.poblacion+idx) << std::endl;
    }
  }
  //os << "Matriz Adyacencias: " << *(g.nodos);
  return os;
}

} // Namespace End
