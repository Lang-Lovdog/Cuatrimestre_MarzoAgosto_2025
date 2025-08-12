#include "MatrizCaracteristicas.hxx"

namespace lovdog{
  GLCMFeatures::GLCMFeatures() : 
    mean       (0),
    variance   (0),
    IDM        (0),
    IDF        (0),
    entropy    (0), 
    ASM        (0),
    energy     (0),
    correlation(0), 
    contrast   (0),
    homogeneity(0) {}

  GLCMFeatures::GLCMFeatures(const GLCMFeatures& d) :
    mean       (d.mean),
    variance   (d.variance),
    IDM        (d.IDM),
    IDF        (d.IDF),
    entropy    (d.entropy),
    ASM        (d.ASM),
    energy     (d.energy),
    correlation(d.correlation),
    contrast   (d.contrast),
    homogeneity(d.homogeneity) {}

  GLRLMFeatures::GLRLMFeatures() :
    SRE(0),
    LRE(0),
    GLN(0),
    RLN(0),
    RP (0) {}

  GLRLMFeatures::GLRLMFeatures(const GLRLMFeatures& d) :
    SRE(d.SRE),
    LRE(d.LRE),
    GLN(d.GLN),
    RLN(d.RLN),
    RP (d.RP) {}

  FeatureMatrix::FeatureMatrix(int rows, int cols, int type) : cv::Mat(rows, cols, type) {}
  FeatureMatrix::FeatureMatrix() : cv::Mat() {}
  FeatureMatrix::FeatureMatrix(const FeatureMatrix& m) : cv::Mat(m) {}
  FeatureMatrix::FeatureMatrix(const cv::Mat& m) : cv::Mat(m) {}
  FeatureMatrix& FeatureMatrix::operator=(const FeatureMatrix& m) { cv::Mat::operator=(m); return *this; }
  FeatureMatrix& FeatureMatrix::operator=(const cv::Mat& m) { cv::Mat::operator=(m); return *this; }
  FeatureMatrix::FeatureMatrix(GLCMFeaturesVector& features){
    this->resize(features.size(), 7);
    for (size_t i = 0; i < features.size(); i++){
      this->at<float>(5, i) = features[i].energy;
      this->at<float>(6, i) = features[i].correlation;
      this->at<float>(4, i) = features[i].contrast;
      this->at<float>(2, i) = features[i].homogeneity;
      this->at<float>(0, i) = features[i].IDF;
      this->at<float>(3, i) = features[i].entropy;
      this->at<float>(1, i) = features[i].variance;
    }
  }
  void FeatureMatrix::add(GLCMFeatures& features){
    this->resize(this->rows + 1, 7);
    this->at<float>(5, this->cols - 1) = features.energy;
    this->at<float>(6, this->cols - 1) = features.correlation;
    this->at<float>(4, this->cols - 1) = features.contrast;
    this->at<float>(2, this->cols - 1) = features.homogeneity;
    this->at<float>(0, this->cols - 1) = features.IDF;
    this->at<float>(3, this->cols - 1) = features.entropy;
    this->at<float>(1, this->cols - 1) = features.variance;
  }
  void FeatureMatrix::rm(size_t index){
    for (size_t j = 0; j < this->rows; j++)
      for (size_t l = 0; l < this->rows; l++){
        if (l == index) continue;
        for (size_t i = index; i < this->cols; i++)
            this->at<float>(j, i) = this->at<float>(l, i);
      }
    this->resize(this->rows - 1, 7);
  }
  std::string FeatureMatrix::csvFormat() const {
    std::string s = "";
    s = "sample, energy, correlation, contrast, homogeneity, IDF, entropy, variance\n";
    for (size_t i = 0; i < this->cols; i++){
      s += std::to_string(i) + ", ";
      for (size_t j = 0; j < this->rows; j++)
        s += std::to_string(this->at<float>(j, i)) + ", ";
      s += "\n";
    }
    return s;
  }



  // Overload the << operator for GLCMFeatures
  std::ostream& operator<<(std::ostream& os, const GLCMFeatures& features) {
      os << "GLCM Features:\n"
         << "  Mean: " << features.mean << "\n"
         << "  Variance: " << features.variance << "\n"
         << "  IDM: " << features.IDM << "\n"
         << "  Entropy: " << features.entropy << "\n"
         << "  ASM: " << features.ASM << "\n"
         << "  Energy: " << features.energy << "\n"
         << "  Correlation: " << features.correlation << "\n"
         << "  Contrast: " << features.contrast << "\n"
         << "  Homogeneity: " << features.homogeneity;
      return os;
  }

  std::ostream& operator<<(std::ostream& os, const GLRLMFeatures& features) {
      os << "GLRLM Features:\n"
         << "  SRE: " << features.SRE << "\n"
         << "  LRE: " << features.LRE << "\n"
         << "  GLN: " << features.GLN << "\n"
         << "  RLN: " << features.RLN << "\n"
         << "  RP: " << features.RP;
      return os;
  }

  void getMaskContours(cv::Mat& Mask, Contours& contours){
    findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  }

  void cropFromMaskContours(cv::Mat &src, Contours& contours, Gallery& dst, float radius, size_t randomCrops){
    cv::Rect rplst, auxRect;
    size_t i,j;
    cv::RNG rng(1980); 

    i=0;
    while(i < contours.size()){
      j=0;
      while(j < randomCrops){
        getReferencePoint(contours[i], rplst);
        calculateCropWindow(rplst, radius, auxRect, rng);
        dst.push_back(src(auxRect));
        ++j;
      }
      ++i;
    }
  }

  void getReferencePoint(Contour& points, cv::Rect& referenceRect){
    referenceRect = cv::boundingRect(points);
  }

  void calculateCropWindow(const cv::Rect& referencePoint, float radius, cv::Rect& cropRect, cv::RNG& rng){

    rng = cv::RNG(rng.uniform(1000,9999)*rng.uniform(200,900)/100);

    float residual[2] = {
      2*radius - referencePoint.width,
      2*radius - referencePoint.height
    };

    cropRect = cv::Rect(
        referencePoint.x-rng.uniform(0,(unsigned int)residual[0]),
        referencePoint.y-rng.uniform(0,(unsigned int)residual[1]),
        2*radius,
        2*radius
    );
  }

  void showGallery(Gallery& gallery,std::string prefix, std::string suffix){
    int image=0;
    std::string Name;
    while((size_t)image < gallery.size()){
      Name = prefix+std::to_string(image+1)+suffix;
      cv::namedWindow(Name, cv::WINDOW_NORMAL);
      cv::imshow(Name, gallery[image]);
      cv::resizeWindow(Name, 300, 300);
      ++image;
    }
  }

  void exportGallery(Gallery& gallery, std::string namePrefix, std::string suffix, std::string extension){
    int image=0;
    while((size_t)image < gallery.size())
      cv::imwrite(namePrefix+std::to_string(image)+suffix+"."+extension, gallery[image++]);
  }


  /* Extracción de características */

  /*GLCM extraction, valid angles are 0, 45, 90, 135 and 180*/
  void graycomatrix(cv::Mat& src, cv::Mat& glcm, uint distance, int angle){
    if(src.empty()) return;
    if(src.channels() != 1) return;
    if(!glcm.empty()) glcm.release();
    if(angle < 0 || angle > 4 ) return;
    cv::Point pi, pf, ps, ss;

    cv::Point direction[5] = {
      cv::Point( distance,        0),
      cv::Point( distance, distance),
      cv::Point(        0, distance),
      cv::Point(-distance, distance),
      cv::Point(-distance,        0)
    };

    glcm = cv::Mat::zeros(256, 256, CV_32FC1);
    //  If angle is 45, 90 or 135, the search will start at d-th row
    pi.x = (angle > GLCMFeatures::ANGLE_0 && angle < GLCMFeatures::ANGLE_180 ) ? distance : 0;
    //  If angle is 135 or 180, the search will start at d-th column
    pi.y = (angle == GLCMFeatures::ANGLE_135 || angle == GLCMFeatures::ANGLE_180 ) ? distance : 0;
    // ps i the  step given by the angle and the distance.
    ps = direction[angle];

    ss = pi;

    while(pi.y < src.rows && (pi+ps).y < src.rows){
      while(pi.x < src.cols && (pf=pi+ps).x < src.cols){
        ++glcm.at<float>( src.at<uchar>(pi), src.at<uchar>(pf));
        ++pi.x;
      } ++pi.y; pi.x = ss.x;
    }
  }

  /* Feature Extraction from GLCM, this computes the energy, correlation, contrast, homogeneity, IDM, entropy, mean and variance */
  void getFeaturesFromGLCM(cv::Mat& glcm, GLCMFeatures& features){
    size_t i,j;
    float p = 0;
    cv::Mat P;
    float mu_i, mu_j, sigma_i, sigma_j;

    features = GLCMFeatures();
    P = glcm / cv::sum(glcm)[0];

    // Compute means (μ_i, μ_j)
    mu_i = mu_j = 0; i = j = 0;
    while (i < (size_t)P.rows) {
      j = 0;
      while (j < (size_t)P.cols) {
        p = P.at<float>(i, j);
        mu_i += i * p;
        mu_j += j * p;
        ++j;
      } ++i;
    }

    // Compute variances (σ_i², σ_j²)
    sigma_i = sigma_j = 0;
    i = 0;
    while (i < (size_t)P.rows) {
      j = 0;
      while (j < (size_t)P.cols) {
        p = P.at<float>(i, j);
        sigma_i += (i - mu_i) * (i - mu_i) * p;
        sigma_j += (j - mu_j) * (j - mu_j) * p;
        ++j;
      } ++i;
    }
    sigma_i = sqrt(sigma_i); sigma_j = sqrt(sigma_j);

    features.mean = 0; i = 0;
    while (i < (size_t)P.rows) {
        j = 0;
        while (j < (size_t)P.cols) {
            features.mean += i * P.at<float>(i, j);  // Row index 'i' as intensity
            ++j;
        }
        ++i;
    }

    // Compute all features in a single pass
    i = 0;
    while (i < (size_t)P.rows) {
      j = 0;
      while (j < (size_t)P.cols) {
        p = P.at<float>(i, j);
        int diff = i - j;
        // ASM (Energy = sqrt(ASM))
        features.ASM += p * p;
        // Contrast (weighted by (i-j)²)
        features.contrast += diff * diff * p;
        // IDM (Homogeneity)
        features.IDM += p / (1 + diff * diff);
        // IDF
        features.IDF += p / (1 + std::abs((float)i-(float)j));
        // Entropy (avoid log(0))
        if (p > 0) features.entropy -= p * log(p) / log(2);
        // Correlation (normalized by σ_i, σ_j)
        features.correlation += ((i - mu_i) * (j - mu_j) * p) / (sigma_i * sigma_j);
        // Variance (σ²)
        features.variance += (i-features.mean) * (i-features.mean) * p;
        // Actualization
        ++j;
      } ++i;
    }
    // Final computations
    features.energy = sqrt(features.ASM);
  }

  void computeGLRLM(const cv::Mat& src, cv::Mat& glrlm, int maxGrayLevel, int direction) {
    if(src.empty()) return;
    if(src.channels() != 1) return;
    if(!glrlm.empty()) glrlm.release();
    if(direction < 0 || direction > 3) return;

    int dx, dy, nx, ny, y, x, runLength;
    uchar currentVal;

    // Directions: 0° (horizontal), 45°, 90° (vertical), 135°
    const int directions[4][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}};
    dx = directions[direction][0];
    dy = directions[direction][1];

    glrlm = cv::Mat::zeros(maxGrayLevel + 1, src.cols + src.rows, CV_32SC1);

    for (y = 0; y < src.rows; ++y) {
      for (x = 0; x < src.cols; ++x) {
        currentVal = src.at<uchar>(y, x);
        runLength = 1;

        // Check bounds and compute run length
        nx = x + dx;
        ny = y + dy;
        while ( nx > -1 && nx < src.cols && ny >= 0 &&
               ny < src.rows && src.at<uchar>(ny, nx) == currentVal) {
          runLength++;
          nx += dx;
          ny += dy;
        }
        glrlm.at<int>(currentVal, runLength - 1)++;
      }
    }
  } 

  void extractGLRLMFeatures(const cv::Mat& glrlm, GLRLMFeatures& features) {
      double totalRuns = cv::sum(glrlm)[0];
      features.RP = totalRuns / (glrlm.rows * glrlm.cols);

      features.SRE = 0;
      features.LRE = 0;
      features.GLN = 0;
      features.RLN = 0;

      for (int i = 0; i < glrlm.rows; ++i) {
          double rowSum = 0;
          for (int j = 0; j < glrlm.cols; ++j) {
              double p = glrlm.at<int>(i, j) / totalRuns;
              if (p > 0) {
                  features.SRE += p / (j + 1);
                  features.LRE += p * (j + 1);
              }
              rowSum += glrlm.at<int>(i, j);
          }
          features.GLN += (rowSum * rowSum) / (totalRuns * totalRuns);
      }

      for (int j = 0; j < glrlm.cols; ++j) {
          double colSum = 0;
          for (int i = 0; i < glrlm.rows; ++i) {
              colSum += glrlm.at<int>(i, j);
          }
          features.RLN += (colSum * colSum) / (totalRuns * totalRuns);
      }
  }

};
