#include "SDH_feat.hxx"
#include <cstdlib>
#include <iostream>
#include <fstream>

namespace lovdog{

  SDH::SDH(){
    this->verbose      = 0;
    this->logOn        = false;
    this->d            = 1;
    this->angle        = SDH::ANGLE_0;
    this->dx           = 1;
    this->dy           = 0;
    this->_mean[0]     =
    this->_mean[1]     =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation  =
    this->contrast     =
    this->homogeneity  =
    this->shadowness   =
    this->prominence   = 
    this->energy       =
    this->entropy      =
    this->mean         =
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
    this->verbose      = 0;
    this->logOn        = false;
    this->d            = 1;
    this->angle        = SDH::ANGLE_0;
    this->dx           = 1;
    this->dy           = 0;
    this->_mean[0]     =
    this->_mean[1]     =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation  =
    this->contrast     =
    this->homogeneity  =
    this->shadowness   =
    this->prominence   = 
    this->energy       =
    this->entropy      =
    this->mean         =
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
    this->verbose      = 0;
    this->logOn        = false;
    this->d            = d;
    this->angle        = angle;
    this->dx           = 1;
    this->dy           = 0;
    this->_mean[0]     =
    this->_mean[1]     =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation  =
    this->contrast     =
    this->homogeneity  =
    this->shadowness   =
    this->prominence   = 
    this->energy       =
    this->entropy      =
    this->mean         =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  SDH::SDH(const SDH& sdh){
    this->verbose      = 0;
    this->logOn        = false;
    this->d            = sdh.d;
    this->angle        = sdh.angle;
    this->dx           = sdh.dx;
    this->dy           = sdh.dy;
    this->_mean[0]     =
    this->_mean[1]     =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation  =
    this->contrast     =
    this->homogeneity  =
    this->shadowness   =
    this->prominence   = 
    this->energy       =
    this->entropy      =
    this->mean         =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = sdh.sumHist[i];
      this->diffHist[i] = sdh.diffHist[i];
    }
  }

  SDH::SDH(size_t d, size_t angle){
    this->verbose      = 0;
    this->logOn        = false;
    this->d            = d;
    this->angle        = angle;
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

    this->_mean[0]     =
    this->_mean[1]     =
    this->_variance[0] =
    this->_variance[1] =
    this->correlation  =
    this->contrast     =
    this->homogeneity  =
    this->shadowness   =
    this->prominence   = 
    this->energy       =
    this->entropy      =
    this->mean         =
    0 ;

    this->sumHist= &Hist[SDH::SUM ][0];
    this->diffHist=&Hist[SDH::DIFF][0];

    for(size_t i = 0; i < 511; i++){
      this->sumHist[i] = 0;
      this->diffHist[i] = 0;
    }
  }

  void SDH::incrementHist(int which_one, int index, double step){
    if(which_one == SDH::DIFF) index+=255;
    if(index > 510) return;
    if(index < 0)   return;
    this->Hist[which_one][index] += step;
  }

  void SDH::decrementHist(int which_one, int index, double step){
    if(which_one == SDH::DIFF) index+=255;
    if(index > 510) return;
    if(index < 0)   return;
    this->Hist[which_one][index] -= step;
  }

  double  SDH::set(const int which_one, int index, double value){
    if(which_one == SDH::DIFF) index+=255;
    if(index < 0)   return -1;
    if(index > 510) return -1;
    this->Hist[which_one][index] = value;
    return this->Hist[which_one][index];
  }

  double  SDH::setRel(const int which_one, int index, double value){
    if(index < 0)   return -1;
    if(index > 510) return -1;
    this->Hist[which_one][index] = value;
    return this->Hist[which_one][index];
  }

  double* SDH::at(const int which_one, int index){
    if(which_one == SDH::DIFF) index+=255;
    if(index < 0)   return nullptr;
    if(index > 510) return nullptr;

    return this->Hist[which_one]+index;
  }

  double* SDH::atRel(const int which_one, size_t index){
    if(index < 0)   return nullptr;
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

  int SDH::getSDH(const cv::Mat& _src, SDH& sdh){
    if(_src.empty()) return -1;
    if(_src.type() != CV_8UC1) return -2;
    if((int)sdh.d > _src.cols-1 || (int)sdh.d > _src.rows-1) return -3;
    cv::Mat src;
    _src.convertTo(src, CV_16SC1);

    cv::Mat Operand1, Operand2, sumMatrix, diffMatrix;
    size_t MatrixSize[2];
    if(sdh.verbose > 1)
      std::cout
        << "Processing Matrix with dimensions: "
        << src.rows << "x" << src.cols << std::endl
        << "Üsing ";
    // Get submatrices
    switch(sdh.angle){
      case SDH::ANGLE_0:
        if(sdh.verbose > 1) std::cout << sdh.d << " X 0" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,      0), cv::Point(src.cols - sdh.d, src.rows        )));
        Operand2 = src(cv::Rect(cv::Point( sdh.d,      0), cv::Point(src.cols        , src.rows        )));
        break;
      case SDH::ANGLE_45:
        if(sdh.verbose > 1) std::cout << sdh.d << " X 45" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,  sdh.d), cv::Point(src.cols - sdh.d, src.rows        )));
        Operand2 = src(cv::Rect(cv::Point( sdh.d,      0), cv::Point(src.cols        , src.rows - sdh.d)));
        break;
      case SDH::ANGLE_90:
        if(sdh.verbose > 1) std::cout << sdh.d << " X 90" << std::endl;
        Operand1 = src(cv::Rect(cv::Point(     0,  sdh.d), cv::Point(src.cols        , src.rows        )));
        Operand2 = src(cv::Rect(cv::Point(     0,      0), cv::Point(src.cols        , src.rows - sdh.d)));
        break;
      case SDH::ANGLE_135:
        if(sdh.verbose > 1) std::cout << sdh.d << " X 135" << std::endl;
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
    if(sdh.verbose > 2){
      std::cout << "Matrix 1: \n" << sumMatrix << std::endl;
      std::cout << "Matrix 2: \n" << diffMatrix << std::endl;
    }

    for (size_t px=0   ; px < 511; ++px) sdh.set(SDH::SUM , px, 0);
    for (size_t px=-255; px < 256; ++px) sdh.set(SDH::DIFF, px, 0);

    for(size_t px=0; px < MatrixSize[0]; ++px) sdh.incrementHist(SDH::SUM , sumMatrix.at<short>(px));
    for(size_t px=0; px < MatrixSize[1]; ++px) sdh.incrementHist(SDH::DIFF,diffMatrix.at<short>(px));

    // Write sdh into a file
    if(sdh.logOn){
      std::fstream file("NoNormSDH.csv", std::ios::out);
      for (size_t bin=0; bin<511; ++bin){
        file << bin << ",  " << *sdh.atRel(SDH::SUM,bin) << ",  " << *sdh.atRel(SDH::DIFF,bin) << std::endl;
      }
      file.close();
    }

    for(size_t bin=0; bin<511; ++bin){
      *sdh.atRel(SDH::SUM,bin)  /= MatrixSize[0];
      *sdh.atRel(SDH::DIFF,bin) /= MatrixSize[1];
    }

    if(sdh.logOn){
      std::fstream file("NoNormSDH.csv", std::ios::out);
      file = std::fstream("SDH.csv", std::ios::out);
      for (size_t bin=0; bin<511; ++bin){
        file << bin << ",  " << *sdh.atRel(SDH::SUM,bin) << ",  " << *sdh.atRel(SDH::DIFF,bin) << std::endl;
      }
      file.close();
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
    sdh._mean[SDH::DIFF] = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh._mean[SDH::DIFF] += bin*(*sdh.at(SDH::DIFF,bin));
    // Mean Sum
    sdh._mean[SDH::SUM]  = 0;
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh._mean[SDH::SUM]  += bin*(*sdh.at(SDH::SUM,bin));
    // Variance Diff
    sdh._variance[SDH::DIFF] = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh._variance[SDH::DIFF] += std::pow(bin-sdh._mean[SDH::DIFF], 2)*(*sdh.at(SDH::DIFF,bin));
    // Variance Sum
    sdh._variance[SDH::SUM]  = 0;
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh._variance[SDH::SUM]  += std::pow(bin-sdh._mean[SDH::SUM] , 2)*(*sdh.at(SDH::SUM ,bin));
    // Mean
    sdh.mean = 0;
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh.mean += bin*(*sdh.at(SDH::SUM,bin));
    sdh.mean /= 2;
    // Correlation
    sdh.correlation = 0;
    sdh.correlation = (sdh._variance[SDH::SUM]-sdh._variance[SDH::DIFF])/
                      (sdh._variance[SDH::SUM]+sdh._variance[SDH::DIFF] + 1e-8);
    // Contrast
    sdh.contrast = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh.contrast    +=  bin*bin**sdh.at(SDH::DIFF,bin);
    // Homogeneity
    sdh.homogeneity = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh.homogeneity +=  (*sdh.at(SDH::DIFF,bin)) / (1+(bin*bin));
    //Energy
    *aux=*(aux+1)=sdh.energy=0;
    for(size_t bin=0; bin<binNumber; ++bin){
      *aux     += (*sdh.atRel(SDH::DIFF,bin))*(*sdh.atRel(SDH::DIFF,bin));
      *(aux+1) += (*sdh.atRel(SDH::SUM ,bin))*(*sdh.atRel(SDH::SUM ,bin));
    }
    sdh.energy = *aux**(aux+1);
    //Entropy
    sdh.entropy = 0;
    for(int bin=0; bin<511; ++bin)
      sdh.entropy -=  (*sdh.atRel(SDH::DIFF,bin))*std::log10(*sdh.atRel(SDH::DIFF,bin)+1e-8) +
                      (*sdh.atRel(SDH::SUM ,bin))*std::log10(*sdh.atRel(SDH::SUM ,bin)+1e-8) ;
    // Shadowness
    sdh.shadowness = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh.shadowness  +=  std::pow(bin-sdh._mean[SDH::DIFF],3)*(*sdh.at(SDH::DIFF,bin));
    // Prominence
    sdh.prominence  = 0;
    for(int bin=-255; bin<256; ++bin)
      sdh.prominence  +=  std::pow(bin-sdh._mean[SDH::DIFF],4)*(*sdh.at(SDH::DIFF,bin));
  }

  void SDH::printFeatures(unsigned int HEADER){
    if(SDH::_MEAN_DIFF     & HEADER)  std::cout << "meanDiff:     " << this->_mean[SDH::DIFF]     << "\n";
    if(SDH::_MEAN_SUM      & HEADER)  std::cout << "meanSum:      " << this->_mean[SDH::SUM]      << "\n";
    if(SDH::_VARIANCE_DIFF & HEADER)  std::cout << "varianceDiff: " << this->_variance[SDH::DIFF] << "\n";
    if(SDH::_VARIANCE_SUM  & HEADER)  std::cout << "varianceSum:  " << this->_variance[SDH::SUM]  << "\n";
    if(SDH::MEAN           & HEADER)  std::cout << "mean:         " << this->mean                 << "\n";
    if(SDH::CORRELATION    & HEADER)  std::cout << "correlation:  " << this->correlation          << "\n";
    if(SDH::CONTRAST       & HEADER)  std::cout << "contrast:     " << this->contrast             << "\n";
    if(SDH::HOMOGENEITY    & HEADER)  std::cout << "homogeneity:  " << this->homogeneity          << "\n";
    if(SDH::ENERGY         & HEADER)  std::cout << "energy:       " << this->energy               << "\n";
    if(SDH::ENTROPY        & HEADER)  std::cout << "entropy:      " << this->entropy              << "\n";
    if(SDH::SHADOWNESS     & HEADER)  std::cout << "shadowness:   " << this->shadowness           << "\n";
    if(SDH::PROMINENCE     & HEADER)  std::cout << "prominence:   " << this->prominence           << "\n";
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
