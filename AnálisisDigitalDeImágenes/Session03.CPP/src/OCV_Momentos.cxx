#include "OCV_Momentos.hxx"
#include "OCV_Course.hxx"

int mainMomentos(int argc, char** argv){
  int blurX,blurY;
  Contornos contornos;
  MomentsVector mv; // vector de momentos de los contornos
  Centros centroides;
  blurX=blurY=12;
  if (argc<2) { 
    std::cout << "Usage " << argv[0]
              << " <image> "
              << "[ -b x y ]"
              << std::endl
  ; return 1; }
  if (argc>2  && !strcmp(argv[2],"-b")){
    std::cout << "Using blur " << argv[3] << "x" << argv[4] << std::endl;
    blurX = atoi(argv[3]);
    blurY = atoi(argv[4]);
  }
  cv::Mat lImage, lImageG, lImageT, lImageC;
  lovdogGetImage(argv[1],lImage);
  
  cvtColor(lImage,lImageG,cv::COLOR_BGR2GRAY);
  blur(lImageG,lImageG,cv::Size(blurX,blurY));
  threshold(lImageG,lImageT,90,200,cv::THRESH_BINARY);

  showIt(lImageT,"Original Image Gray Threshold",false);

  lovdogExtraeContornos(lImageT, contornos);
  lovdogMomentos(contornos, mv);
  lovdogCentroides(mv, centroides);

  printCentroides(centroides, mv, contornos);

  lImageC = cv::Mat::zeros(lImageT.size(), CV_8UC3);
  drawCentroid(centroides, contornos, lImageC);
  showIt(lImageC,"Centroides",false);

  cv::waitKey(0);
  return 0;
}

int drawCentroid(const Centros& centroides, const Contornos& src, cv::Mat& dstImg){
  // Random color scalar for each contour
  cv::RNG rng(1673);
  for(size_t i=0; i<src.size(); ++i){
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::drawContours(dstImg,src,(int)i,color,1);
    cv::circle(dstImg, centroides[i], 4, color, -1);
  }
  return 0;
}

int lovdogMomentos(const Contornos& src, MomentsVector& mv){
  for(size_t i=0; i<src.size(); ++i)
    mv.push_back(cv::moments(src[i]));
  return 0;
}

int lovdogCentroides(const MomentsVector& mv, Centros& centroides){
  for(size_t i=0; i< mv.size(); ++i)
    centroides.push_back(cv::Point(
        mv[i].m10/(mv[i].m00 + 1e-5),
        mv[i].m01/(mv[i].m00 + 1e-5)
    ));
  return 0;
}

int printCentroides(const Centros& centroides, const MomentsVector& mv, const Contornos& src){
  for(size_t i=0; i< centroides.size(); ++i)
    std::cout  << " * Contour ["       << i             << "]: "
               << "Centroid = mc["     << centroides[i] << "], "
               << "Area (M_00) = "     << std::fixed    << std::setprecision(2) << mv[i].m00
               << std::endl
               << "\t\t Area (OpenCV): "   << cv::contourArea(src[i])       << ", "
               << " Lenght (OpenCV): " << cv::arcLength(src[i],true)
               << std::endl
    ;
  return 0;
}
