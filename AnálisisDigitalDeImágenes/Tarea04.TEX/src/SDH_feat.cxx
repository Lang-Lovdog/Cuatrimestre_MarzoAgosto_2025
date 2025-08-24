#include "SDH_feat.hxx"
#include <cstdlib>
#include <iostream>
#include <fstream>

namespace lovdog{

  SDH::SDH(){
    this->d = 1;
    this->angle = SDH::ANGLE_0;
    this->dx = 1;
    this->dy = 0;
    this->_mean[0]    =
    this->_mean[1]    =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation =
    this->contrast    =
    this->homogeneity =
    this->shadowness  =
    this->prominence  = 
    this->energy      =
    this->entropy     =
    this->mean        =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  SDH::SDH(const cv::Mat& src){
    src.copyTo(this->src);
    this->d = 1;
    this->angle = SDH::ANGLE_0;
    this->dx = 1;
    this->dy = 0;
    this->_mean[0]    =
    this->_mean[1]    =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation =
    this->contrast    =
    this->homogeneity =
    this->shadowness  =
    this->prominence  = 
    this->energy      =
    this->entropy     =
    this->mean        =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  SDH::SDH(const cv::Mat& src, uint d, uint angle){
    src.copyTo(this->src);
    this->d = d;
    this->angle = angle;
    this->dx = 1;
    this->dy = 0;
    this->_mean[0]    =
    this->_mean[1]    =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation =
    this->contrast    =
    this->homogeneity =
    this->shadowness  =
    this->prominence  = 
    this->energy      =
    this->entropy     =
    this->mean        =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  SDH::SDH(const SDH& sdh){
    this->d = sdh.d;
    this->angle = sdh.angle;
    this->dx = sdh.dx;
    this->dy = sdh.dy;

    this->_mean[0]    =
    this->_mean[1]    =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation =
    this->contrast    =
    this->homogeneity =
    this->shadowness  =
    this->prominence  = 
    this->energy      =
    this->entropy     =
    this->mean        =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = sdh.sumHist[i];
      this->diffHist[i] = sdh.diffHist[i];
    }
  }

  SDH::SDH(size_t d, size_t angle){
    this->d = d;
    this->angle = angle;
    switch(angle){
      case SDH::ANGLE_0:
        this->dx =  1;
        this->dy =  0;
        break;
      case SDH::ANGLE_45:
        this->dx =  1;
        this->dy =  1;
        break;
      case SDH::ANGLE_90:
        this->dx =  0;
        this->dy = -1;
        break;
      case SDH::ANGLE_135:
        this->dx = -1;
        this->dy = -1;
        break;
    }

    this->_mean[0]    =
    this->_mean[1]    =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation =
    this->contrast    =
    this->homogeneity =
    this->shadowness  =
    this->prominence  = 
    this->energy      =
    this->entropy     =
    this->mean        =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  double* SDH::at(const int which_one, int index){
    switch(which_one){
      case SDH::DIFF:
        index+=255; if(index > 510) return nullptr;
        return this->diffHist+index;
      case SDH::SUM:
        if(index > 510) return nullptr;
        return this->sumHist+index;
    }
    return nullptr;
  }

  double* SDH::atRel(const int which_one, size_t index){
    if(index > 510) return nullptr;
    return &this->Hist[which_one][index];
  }

  int SDH::getSDH(void){
    if(this->src.empty()) return -1;
    if(this->src.type() != CV_8UC1) return -2;
    if((int)this->d > this->src.cols-1 || (int)this->d > this->src.rows-1) return -3;
    return SDH::getSDH(this->src, *this);
  }

  int SDH::getSDH(const cv::Mat& src){
    if(src.empty()) return -1;
    if(src.type() != CV_8UC1) return -2;
    if((int)this->d > this->src.cols-1 || (int)this->d > src.rows-1) return -3;
    src.copyTo(this->src);
    return SDH::getSDH(this->src, *this);
  }

  int SDH::getSDH(uint d, uint angle){
    if(this->src.empty()) return -1;
    if(this->src.type() != CV_8UC1) return -2;
    if((int)d > this->src.cols-1 || (int)d > this->src.rows-1) return -3;
    this->d=d;
    this->angle=angle;
    return SDH::getSDH(this->src, *this);
  }

  int SDH::getSDH(const cv::Mat& src, uint d, uint angle){
    if(src.empty()) return -1;
    if(src.type() != CV_8UC1) return -2;
    if((int)d > src.cols-1 || (int)d > src.rows-1) return -3;
    src.copyTo(this->src);
    this->d = d;
    this->angle = angle;
    return SDH::getSDH(this->src, *this);
  }

  int SDH::getSDH(const cv::Mat& src, SDH& sdh){
    if(src.empty()) return -1;
    if(src.type() != CV_8UC1) return -2;
    if((int)sdh.d > src.cols-1 || (int)sdh.d > src.rows-1) return -3;
    cv::Mat Operand1, Operand2, sumMatrix, diffMatrix;
    size_t MatrixSize[2];
    std::cout
      << "Processing Matrix with dimensions: "
      << src.rows << "x" << src.cols << std::endl
      << "Üsing ";
    // Get submatrices
    switch(sdh.angle){
      case SDH::ANGLE_0:
        std::cout << sdh.d << " X 0" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,      0), cv::Point(src.cols - sdh.d, src.rows        )));
        Operand2 = src(cv::Rect(cv::Point( sdh.d,      0), cv::Point(src.cols        , src.rows        )));
        break;
      case SDH::ANGLE_45:
        std::cout << sdh.d << " X 45" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,  sdh.d), cv::Point(src.cols - sdh.d, src.rows        )));
        Operand2 = src(cv::Rect(cv::Point( sdh.d,      0), cv::Point(src.cols        , src.rows - sdh.d)));
        break;
      case SDH::ANGLE_90:
        std::cout << sdh.d << " X 90" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,  sdh.d), cv::Point(src.cols        , src.rows        )));
        Operand2 = src(cv::Rect(cv::Point(     0,      0), cv::Point(src.cols        , src.rows - sdh.d)));
        break;
      case SDH::ANGLE_135:
        std::cout << sdh.d << " X 135" << std::endl;
        Operand1 = src(cv::Rect(cv::Point( sdh.d,  sdh.d), cv::Point(src.cols        , src.rows        )));
        Operand2 = src(cv::Rect(cv::Point(     0,      0), cv::Point(src.cols - sdh.d, src.rows - sdh.d)));
        break;
    }
    // Los rangos van de 0 a 255 para cada Operando
    // El rango de valores de diffMatrix es de -255 a 255
    // El rango de valores de sumMatrix es de 0 a 510
    // En ambos casos, existe un conjunto de 511 posibles valores
    // Debido a que los índices en c no pueden ser negativos,
    // se procede con un artificio para el rango de valores dentro de la clase SDH
     sumMatrix = Operand1 + Operand2;
    diffMatrix = Operand1 - Operand2;

    MatrixSize[0] =  sumMatrix.rows*sumMatrix.cols;
    MatrixSize[1] = diffMatrix.rows*diffMatrix.cols;
    for(size_t px=0; px < MatrixSize[0]; ++px) ++(*sdh.at(SDH::SUM,sumMatrix.data[px]));
    for(size_t px=0; px < MatrixSize[1]; ++px) ++(*sdh.at(SDH::DIFF,diffMatrix.data[px]));

    for(size_t bin=0; bin<511; ++bin){
      *sdh.atRel(SDH::SUM,bin)  /= MatrixSize[0];
      *sdh.atRel(SDH::DIFF,bin) /= MatrixSize[1];
    }

    return 0;
  }

  void SDH::computeFeatures(void){
    SDH::computeFeatures(*this);
  }

  void SDH::computeFeatures(SDH &sdh){
    uint binNumber = 511;
    float aux[2];
    // Mean Diff
    for(int bin=-255; bin<256; ++bin)
      sdh._mean[SDH::DIFF] += bin*(*sdh.at(SDH::DIFF,bin));
    // Mean Sum
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh._mean[SDH::SUM]  += bin*(*sdh.at(SDH::SUM,bin));
    // Variance Diff
    for(int bin=-255; bin<256; ++bin)
      sdh._variance[SDH::DIFF] += std::pow(bin-sdh._mean[SDH::DIFF], 2)*(*sdh.at(SDH::DIFF,bin));
    // Variance Sum
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh._variance[SDH::SUM]  += std::pow(bin-sdh._mean[SDH::SUM] , 2)*(*sdh.at(SDH::SUM ,bin));
    // Mean
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh.mean += bin*(*sdh.at(SDH::SUM,bin));
    sdh.mean /= 2;
    // Correlation
    sdh.correlation = (sdh._variance[SDH::SUM]-sdh._variance[SDH::DIFF])/
                      (sdh._variance[SDH::SUM]+sdh._variance[SDH::DIFF] + 1e-8);
    // Contrast
    sdh.contrast    =  sdh._variance[SDH::DIFF];
    // Homogeneity
    for(int bin=-255; bin<256; ++bin)
      sdh.homogeneity +=  (*sdh.at(SDH::DIFF,bin)) / (1+(bin*bin));
    //Energy
    *aux=0;
    for(size_t bin=0; bin<binNumber; ++bin){
      *aux     += (*sdh.atRel(SDH::DIFF,bin))*(*sdh.atRel(SDH::DIFF,bin));
      *(aux+1) += (*sdh.atRel(SDH::SUM ,bin))*(*sdh.atRel(SDH::SUM ,bin));
    }
    sdh.energy = *aux**(aux+1);
    //Entropy
    for(int bin=0; bin<511; ++bin)
      sdh.entropy -=  (*sdh.atRel(SDH::DIFF,bin))*std::log2(*sdh.atRel(SDH::DIFF,bin)+1e-8) +
                      (*sdh.atRel(SDH::SUM ,bin))*std::log2(*sdh.atRel(SDH::SUM ,bin)+1e-8) ;
    // Shadowness
    for(int bin=-255; bin<256; ++bin)
      sdh.shadowness  +=  std::pow(bin-sdh._mean[SDH::DIFF],3)*(*sdh.at(SDH::DIFF,bin));
    // Prominence
    for(int bin=-255; bin<256; ++bin)
      sdh.prominence  +=  std::pow(bin-sdh._mean[SDH::DIFF],4)*(*sdh.at(SDH::DIFF,bin));
  }

  void SDH::printFeatures(void){
    std::cout
         << "meanDiff:     " << this->_mean[SDH::DIFF]     << "\n"
         << "meanSum:      " << this->_mean[SDH::SUM]      << "\n"
         << "varianceDiff: " << this->_variance[SDH::DIFF] << "\n"
         << "varianceSum:  " << this->_variance[SDH::SUM]  << "\n"
         << "mean:         " << this->mean                 << "\n"
         << "correlation:  " << this->correlation          << "\n"
         << "contrast:     " << this->contrast             << "\n"
         << "homogeneity:  " << this->homogeneity          << "\n"
         << "energy:       " << this->energy               << "\n"
         << "entropy:      " << this->entropy              << "\n"
         << "shadowness:   " << this->shadowness           << "\n"
         << "prominence:   " << this->prominence           << "\n"
    ;
  }

  void SDH::toCSV(std::vector<SDH>& sdhs, std::string filename, unsigned int HEADER){
    std::fstream file(filename, std::ios::out);
    if(SDH::_MEAN_DIFF     & HEADER)  file << "meanDiff,";
    if(SDH::_MEAN_SUM      & HEADER)  file << "meanSum,";
    if(SDH::_VARIANCE_DIFF & HEADER)  file << "varianceDiff,";
    if(SDH::_VARIANCE_SUM  & HEADER)  file << "varianceSum,";
    if(SDH::MEAN           & HEADER)  file << "mean,";
    if(SDH::CORRELATION    & HEADER)  file << "correlation,";
    if(SDH::CONTRAST       & HEADER)  file << "contrast,";
    if(SDH::HOMOGENEITY    & HEADER)  file << "homogeneity,";
    if(SDH::ENERGY         & HEADER)  file << "energy,";
    if(SDH::ENTROPY        & HEADER)  file << "entropy,";
    if(SDH::SHADOWNESS     & HEADER)  file << "shadowness,";
    if(SDH::PROMINENCE     & HEADER)  file << "prominence\n";

    for (auto &sdh : sdhs){
      if(SDH::_MEAN_DIFF     & HEADER)  file << sdh._mean[SDH::DIFF]      << ",";
      if(SDH::_MEAN_SUM      & HEADER)  file << sdh._mean[SDH::SUM]       << ",";
      if(SDH::_VARIANCE_DIFF & HEADER)  file << sdh._variance[SDH::DIFF]  << ",";
      if(SDH::_VARIANCE_SUM  & HEADER)  file << sdh._variance[SDH::SUM]   << ",";
      if(SDH::MEAN           & HEADER)  file << sdh.mean                  << ",";
      if(SDH::CORRELATION    & HEADER)  file << sdh.correlation           << ",";
      if(SDH::CONTRAST       & HEADER)  file << sdh.contrast              << ",";
      if(SDH::HOMOGENEITY    & HEADER)  file << sdh.homogeneity           << ",";
      if(SDH::ENERGY         & HEADER)  file << sdh.energy                << ",";
      if(SDH::ENTROPY        & HEADER)  file << sdh.entropy               << ",";
      if(SDH::SHADOWNESS     & HEADER)  file << sdh.shadowness            << ",";
      if(SDH::PROMINENCE     & HEADER)  file << sdh.prominence                  ;
      file << "\n";
    }
    file.close();
  }

  void SDH::toCSV_WriteHeader(std::string filename, unsigned int HEADER){
    std::cout << "Writing header to" << filename << std::endl;
    std::fstream file;
    file.open(filename, std::ios::out);
    file << "Sample,";
    if(SDH::_MEAN_DIFF     & HEADER)  file << "meanDiff,"     ;
    if(SDH::_MEAN_SUM      & HEADER)  file << "meanSum,"      ;
    if(SDH::_VARIANCE_DIFF & HEADER)  file << "varianceDiff," ;
    if(SDH::_VARIANCE_SUM  & HEADER)  file << "varianceSum,"  ;
    if(SDH::MEAN           & HEADER)  file << "mean,"         ;
    if(SDH::CORRELATION    & HEADER)  file << "correlation,"  ;
    if(SDH::CONTRAST       & HEADER)  file << "contrast,"     ;
    if(SDH::HOMOGENEITY    & HEADER)  file << "homogeneity,"  ;
    if(SDH::ENERGY         & HEADER)  file << "energy,"       ;
    if(SDH::ENTROPY        & HEADER)  file << "entropy,"      ;
    if(SDH::SHADOWNESS     & HEADER)  file << "shadowness,"   ;
    if(SDH::PROMINENCE     & HEADER)  file << "prominence"    ;
    file << std::endl;
    file.close();
  }

  void SDH::toCSV(std::string filename, const uint HEADER, bool append, std::string name){
    std::cout << "Writing to " << filename << std::endl;
    std::fstream file;
    if(!append){
      file.open(filename, std::ios::out);
      file << "Sample,";
      if(SDH::_MEAN_DIFF     & HEADER)  file << "meanDiff,"     ;
      if(SDH::_MEAN_SUM      & HEADER)  file << "meanSum,"      ;
      if(SDH::_VARIANCE_DIFF & HEADER)  file << "varianceDiff," ;
      if(SDH::_VARIANCE_SUM  & HEADER)  file << "varianceSum,"  ;
      if(SDH::MEAN           & HEADER)  file << "mean,"         ;
      if(SDH::CORRELATION    & HEADER)  file << "correlation,"  ;
      if(SDH::CONTRAST       & HEADER)  file << "contrast,"     ;
      if(SDH::HOMOGENEITY    & HEADER)  file << "homogeneity,"  ;
      if(SDH::ENERGY         & HEADER)  file << "energy,"       ;
      if(SDH::ENTROPY        & HEADER)  file << "entropy,"      ;
      if(SDH::SHADOWNESS     & HEADER)  file << "shadowness,"   ;
      if(SDH::PROMINENCE     & HEADER)  file << "prominence"    ;
      file << std::endl;
      file << name << "_" << this->d <<"x" << this->angle  <<",";
      if(SDH::_MEAN_DIFF     & HEADER)  file << this->_mean[SDH::DIFF]      << ",";
      if(SDH::_MEAN_SUM      & HEADER)  file << this->_mean[SDH::SUM]       << ",";
      if(SDH::_VARIANCE_DIFF & HEADER)  file << this->_variance[SDH::DIFF]  << ",";
      if(SDH::_VARIANCE_SUM  & HEADER)  file << this->_variance[SDH::SUM]   << ",";
      if(SDH::MEAN           & HEADER)  file << this->mean                  << ",";
      if(SDH::CORRELATION    & HEADER)  file << this->correlation           << ",";
      if(SDH::CONTRAST       & HEADER)  file << this->contrast              << ",";
      if(SDH::HOMOGENEITY    & HEADER)  file << this->homogeneity           << ",";
      if(SDH::ENERGY         & HEADER)  file << this->energy                << ",";
      if(SDH::ENTROPY        & HEADER)  file << this->entropy               << ",";
      if(SDH::SHADOWNESS     & HEADER)  file << this->shadowness            << ",";
      if(SDH::PROMINENCE     & HEADER)  file << this->prominence                  ;
      file << std::endl;
      file.close();
      return;
    } 
    file.open(filename, std::ios::out | std::ios::app);
    file << name << "_" << this->d <<"x" << this->angle  <<",";
    if(SDH::_MEAN_DIFF     & HEADER)  file << this->_mean[SDH::DIFF]      << ",";
    if(SDH::_MEAN_SUM      & HEADER)  file << this->_mean[SDH::SUM]       << ",";
    if(SDH::_VARIANCE_DIFF & HEADER)  file << this->_variance[SDH::DIFF]  << ",";
    if(SDH::_VARIANCE_SUM  & HEADER)  file << this->_variance[SDH::SUM]   << ",";
    if(SDH::MEAN           & HEADER)  file << this->mean                  << ",";
    if(SDH::CORRELATION    & HEADER)  file << this->correlation           << ",";
    if(SDH::CONTRAST       & HEADER)  file << this->contrast              << ",";
    if(SDH::HOMOGENEITY    & HEADER)  file << this->homogeneity           << ",";
    if(SDH::ENERGY         & HEADER)  file << this->energy                << ",";
    if(SDH::ENTROPY        & HEADER)  file << this->entropy               << ",";
    if(SDH::SHADOWNESS     & HEADER)  file << this->shadowness            << ",";
    if(SDH::PROMINENCE     & HEADER)  file << this->prominence                  ;
    file << std::endl;
    file.close();
  }

};
