#include "Detect.hxx"
#include "OCV_Course.hxx"
#include <vector>

int mainDetectCirc(int argc, char** argv){
  if (argc<2) { 
    std::cout << "Usage " << argv[0]
              << " <image> "
              << std::endl
  ; return 1; }
  cv::Mat lImage, lImageG, lImageT, lImageC;
  if(!lovdogGetImage(argv[1],lImage)) return 1;
  
  cvtColor(lImage,lImageG,cv::COLOR_BGR2GRAY);

  showIt(lImageG,"Original Image Gray",false);

  cv::Canny(lImageG,lImageT,30,200);

  showIt(lImageT,"Edges",false);

  cvtColor(lImageT,lImageC,cv::COLOR_GRAY2BGR);

  std::vector<cv::Vec3f> Circle;
  cv::HoughCircles(
      lImageT, Circle,
      cv::HOUGH_GRADIENT,
      1,
      lImageT.rows/16,
      30,
      200,
      500,
      1000
  );

  lovdogDibujaCirculos(Circle, lImageC);

  showIt(lImageC,"Rectas",false);

  cv::waitKey(0);
  return 0;
}

int mainDetect(int argc, char** argv){
  if (argc<2) { 
    std::cout << "Usage " << argv[0]
              << " <image> "
              << std::endl
  ; return 1; }
  cv::Mat lImage, lImageG, lImageT, lImageC;
  if(!lovdogGetImage(argv[1],lImage)) return 1;
  
  cvtColor(lImage,lImageG,cv::COLOR_BGR2GRAY);

  showIt(lImageG,"Original Image Gray",false);

  cv::Canny(lImageG,lImageT,90,200);

  showIt(lImageT,"Edges",false);

  cvtColor(lImageT,lImageC,cv::COLOR_GRAY2BGR);

  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(
      lImageT, linesP,
      1,
      CV_PI/180,
      20,
      40,
      10
  );

  lovdogDibujaLineas(linesP, lImageC);

  showIt(lImageC,"Rectas",false);

  cv::waitKey(0);
  return 0;
}

int lovdogDibujaCirculos(std::vector<cv::Vec3f> Circle, cv::Mat &lImageC){
  cv::RNG rng(1673);
  cv::Scalar color;
  cv::Point Centro;
  int radio;
  for (size_t i = 0; i < Circle.size(); i++) {
    color= cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
    Centro=cv::Point(Circle[i][0], Circle[i][1]);
    radio=Circle[i][2];
    cv::circle(
        lImageC,
        Centro, radio,
        color,
        1, cv::LINE_AA
    );
    cv::circle(
        lImageC,
        Centro, 1,
        color,
        1, cv::LINE_AA
    );
  }
  return 0;
}

int lovdogDibujaLineas(std::vector<cv::Vec4i> linesP, cv::Mat &lImageC){
  cv::RNG rng(1673);
  cv::Scalar color;
  cv::Point P1, P2;
  for (size_t i = 0; i < linesP.size(); i++) {
    color= cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
    P1=cv::Point(linesP[i][0], linesP[i][1]);
    P2=cv::Point(linesP[i][2], linesP[i][3]);
    cv::line(
        lImageC,
        P1, P2,
        color,
        3, cv::LINE_AA
    );
  }
  return 0;
}
