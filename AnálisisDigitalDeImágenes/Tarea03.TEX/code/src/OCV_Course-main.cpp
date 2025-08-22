#include <iostream>
#include "OCV_Course.hxx"
#include "OCV_Momentos.hxx"

int mainMomentClassifier(int argc, char** argv);
int getSimilitude(const double* mi,const double mc[6][7], double* S);

int main (int argc, char* argv[]){
  mainMomentClassifier(argc,argv);
  return 0;
}

int mainMomentClassifier(int argc, char** argv){
  int blurX,blurY;
  Contornos contornos;
  MomentsVector mv; // vector de momentos de los contornos
  Centros centroides;

  blurX=blurY=8;
  if (argc<2) { 
    std::cout << "Usage " << argv[0]
              << " <image> "
              << "[ -b x y ]"
              << std::endl
  ; return 1; }
  if (argc>2  && !strcmp(argv[2],"-b"))
    std::cout << "Using blur " << (blurX=atoi(argv[3]))<< "x" << (blurY=atoi(argv[4])) << std::endl;

  cv::Mat lImage, lImageG, lImageT;
  double
    Prototipos[6][7] = {
      {0.49  ,  2.00  ,  1.96  , 3.79   , 6.82    ,  4.95  , -6.80 }  ,
      {0.22  ,  0.49  ,  2.51  ,  3.05  ,  5.92   ,  3.59  , -6.05 }  ,
      {0.55  ,  3.08  ,  4.90  ,  5.35  , -10.64  ,  6.90  , -10.61}  ,
      {0.44  ,  1.55  ,  2.52  ,  3.76  ,  7.21   ,  4.96  ,  6.95 }  ,
      {0.42  ,  1.40  ,  2.72  ,  4.13  ,  7.55   ,  4.84  ,  9.07 }  , 
      {0.46  ,  1.74  ,  1.89  ,  2.59  , -4.88   , -3.49  , -5.15 }} ,
    S[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    Hu[7]= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t minval = 0;

  lovdogGetImage(argv[1],lImage);

  if(lImage.empty()) { std::cout << "Error File not Found" << std::endl; return 1; }
  
  cvtColor(lImage,lImageG,cv::COLOR_BGR2GRAY);
  blur(lImageG,lImageG,cv::Size(blurX,blurY));
  threshold(lImageG,lImageT,90,200,cv::THRESH_BINARY);

  lovdogExtraeContornos(lImageT, contornos);
  lovdogMomentos(contornos, mv);
  lovdogCentroides(mv, centroides);

  for(size_t c=0; c<contornos.size(); ++c){
    cv::HuMoments(mv[c], Hu);
    // Compute log of hu moments
    for(size_t i=0; i<7; ++i) Hu[i] = -1 * std::copysign(1.0, Hu[i])*std::log10(std::abs(Hu[i]));

    getSimilitude(Hu, Prototipos, S);

    std::cout << "Similitud listada a la clase del contorno " << c+1 << " (valor más pequeño, similitud mayor):" << std::endl;
    for(size_t i=0; i<6; ++i) std::cout << "S[" << i+1 << "] = " <<  S[i] << std::endl;
    minval = std::min_element(S, S+6)-S+1;
    std::cout << "Clase más similiar: " << minval << std::endl;
  }

  return 0;
}

int getSimilitude(const double* mi,const double mc[6][7], double* S){
  for(int i=0; i<6; ++i)
    for(int j=0; j<7; ++j)
      S[i] += std::abs(mi[j]-mc[i][j]);
  return 0;
}

