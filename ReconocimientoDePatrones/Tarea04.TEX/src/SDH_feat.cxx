#include "SDH_feat.hxx"

namespace lovdog{

  SDH::SDH(){
    d = 1;
    angle = SDH::ANGLE_0;
    dx = 1;
    dy = 0;
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
    switch(which_one){
      case SDH::DIFF:
        if(index > 510) return nullptr;
        return this->diffHist+index;
      case SDH::SUM:
        if(index > 510) return nullptr;
        return this->sumHist+index;
    }
    return nullptr;
  }

  int SDH::getSDH(const cv::Mat& src, SDH& sdh){
    if(src.empty()) return -1;
    if(src.type() != CV_8UC1) return -2;
    if(sdh.d > src.cols-1 || sdh.d > src.rows-1) return -3;
    cv::Mat Operand1, Operand2, sumMatrix, diffMatrix;
    size_t MatrixSize[2];
    // Get submatrices
    switch(sdh.angle){
      case SDH::ANGLE_0:
        Operand1 = src(cv::Rect(     0,      0, src.cols - sdh.d, src.rows        ));
        Operand2 = src(cv::Rect( sdh.d,      0, src.cols        , src.rows        ));
        break;
      case SDH::ANGLE_45:
        Operand1 = src(cv::Rect(     0,  sdh.d, src.cols - sdh.d, src.rows        ));
        Operand2 = src(cv::Rect( sdh.d,      0, src.cols        , src.rows - sdh.d));
        break;
      case SDH::ANGLE_90:
        Operand1 = src(cv::Rect(     0,  sdh.d, src.cols        , src.rows        ));
        Operand2 = src(cv::Rect(     0,      0, src.cols        , src.rows - sdh.d));
        break;
      case SDH::ANGLE_135:
        Operand1 = src(cv::Rect( sdh.d,  sdh.d, src.cols        , src.rows        ));
        Operand2 = src(cv::Rect(     0,      0, src.cols - sdh.d, src.rows - sdh.d));
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

    MatrixSize[0] = sumMatrix.rows*sumMatrix.cols;
    MatrixSize[1] = diffMatrix.rows*diffMatrix.cols;
    for(size_t px=0; px < MatrixSize[0]; ++px) ++(*sdh.at(SDH::SUM,sumMatrix.data[px]));
    for(size_t px=0; px < MatrixSize[1]; ++px) ++(*sdh.at(SDH::DIFF,diffMatrix.data[px]));

    for(size_t bin=0; bin<511; ++bin){
      *sdh.atRel(SDH::SUM,bin)  /= MatrixSize[0];
      *sdh.atRel(SDH::DIFF,bin) /= MatrixSize[1];
    }

    return 0;
  }

  void SDH::computeFeatures(SDH &sdh){
    uint binNumber = 511;
    // Mean Diff
    for(int bin=-255; bin<256; ++bin)
      sdh.mean[SDH::DIFF] += bin*(*sdh.at(SDH::DIFF,bin));
    // Mean Sum
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh.mean[SDH::SUM]  += bin*(*sdh.at(SDH::SUM,bin));
    // Variance Diff
    for(int bin=-255; bin<256; ++bin)
      sdh.variance[SDH::DIFF] += std::pow(bin-sdh.mean[SDH::DIFF], 2)*(*sdh.at(SDH::DIFF,bin));
    // Variance Sum
    for(size_t bin=0; bin<binNumber; ++bin)
      sdh.variance[SDH::SUM]  += std::pow(bin-sdh.mean[SDH::SUM] , 2)*(*sdh.at(SDH::SUM ,bin));
    // Correlation
    sdh.correlation = (sdh.variance[SDH::SUM]-sdh.variance[SDH::DIFF])/
                      (sdh.variance[SDH::SUM]+sdh.variance[SDH::DIFF] + 1e-8);
    // Contrast
    sdh.contrast    =  sdh.variance[SDH::DIFF];
    // Homogeneity
    for(int bin=-255; bin<256; ++bin)
      sdh.homogeneity +=  (*sdh.at(SDH::DIFF,bin)) / (1+std::abs(bin));
    // Shadowness
    for(int bin=-255; bin<256; ++bin)
      sdh.shadowness  +=  std::pow(bin-sdh.mean[SDH::DIFF],3)*(*sdh.at(SDH::DIFF,bin));
    // Prominence
    for(int bin=-255; bin<256; ++bin)
      sdh.prominence  +=  std::pow(bin-sdh.mean[SDH::DIFF],4)*(*sdh.at(SDH::DIFF,bin));
  }

  void SDH::toCSV(SDHs &sdhs, std::string filename){
    std::fstream file(filename, std::ios::out);
    file
         << "meanDiff,"
         << "meanSum,"
         << "varianceDiff,"
         << "varianceSum,"
         << "correlation,"
         << "contrast,"
         << "homogeneity,"
         << "shadowness,"
         << "prominence\n"
    ;
    for (auto &sdh : sdhs)
      file 
           << sdh.mean[SDH::DIFF]     << ","
           << sdh.mean[SDH::SUM]      << ","
           << sdh.variance[SDH::DIFF] << ","
           << sdh.variance[SDH::SUM]  << ","
           << sdh.correlation         << ","
           << sdh.contrast            << ","
           << sdh.homogeneity         << ","
           << sdh.shadowness          << ","
           << sdh.prominence          << "\n"
      ;
    file.close();
  }

};
