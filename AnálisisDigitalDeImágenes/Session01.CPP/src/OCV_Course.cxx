#include <iostream>
#include <core.hpp>
#include <imgproc.hpp>
#include <opencv.hpp>
#include <highgui.hpp>
#include "OCV_Course.hxx"

int mainHolaMundo (int argc,char* argv[]){
  if(argc<2) { std::cout << "Requires an argument (image)"; return 1; }
  std::cout << "Checking OpenCV installation in PC /opencv/ \n";
  cv::Mat Input,InputClone,InputCopy;

  Input      = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);

  if(Input.empty()) { std::cout << "Image reading error"; return 1; }


  InputClone = Input.clone();
  Input.copyTo(InputCopy);


  imshow("Input image"                           , Input);
  imshow("Input image 2 (The Clone Attack)"      , InputClone);
  imshow("Input image 3 (The Rising of the Copy)", InputCopy);


  imwrite("FotoRobadaUwU.png",InputCopy);

  cv::waitKey(0);
  return 0;
}

int MatBasics(void){
  std::cout << "Basic Mat object arithmetics\n";
  std::cout << "Matrix Creation\n";
  std::cout << "Zeros Matrix (2,3)\n";
  cv::Mat matrix_zeros;
  matrix_zeros = cv::Mat::zeros(2,3,CV_32F);
  std::cout << matrix_zeros
            << std::endl << std::endl
  ;

  std::cout << "Ones Matrix (5,5)\n";
  cv::Mat matrix_ones;
  matrix_ones = cv::Mat::ones(5,5,CV_32F);
  std::cout << matrix_ones
            << std::endl << std::endl
  ;

  std::cout << "Identity Matrix (5,5)\n";
  cv::Mat matrix_identity;
  matrix_identity = cv::Mat::eye(5,5,CV_32F);
  std::cout << matrix_identity
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 32F\n";
  cv::Mat matrix_randomf = cv::Mat(2,4,CV_32F);
  randu(matrix_randomf,2,8);
  std::cout << matrix_randomf
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 8U\n";
  cv::Mat matrix_randomc = cv::Mat(2,4,CV_8U);
  randu(matrix_randomc,0,255);
  std::cout << matrix_randomc
            << std::endl << std::endl
  ;

  std::cout << "Random Matrix (2,4) 8UC3\n";
  cv::Mat matrix_randomc3 = cv::Mat(2,4,CV_8UC3);
  randu(matrix_randomc3,0,255);
  std::cout << matrix_randomc3
            << std::endl << std::endl
  ;

  return 0;
}

int MatricOperations(){
  cv::Mat A = cv::Mat::eye ( cv::Size(3,2), CV_32F );
  cv::Mat B = cv::Mat::ones( cv::Size(3,2), CV_32F );
  cv::Mat C = cv::Mat::eye ( cv::Size(3,3), CV_32F );
  cv::Mat D = cv::Mat::ones( cv::Size(3,3), CV_32F );
  cv::Mat E = cv::Mat::eye ( cv::Size(2,3), CV_32F );
  std::cout<< "Given the matrices " << std::endl
           << "Matrix A"                << std::endl << A                 << std::endl << std::endl
           << "Matrix B"                << std::endl << B                 << std::endl << std::endl
           << "Matrix C"                << std::endl << C                 << std::endl << std::endl
           << "Matrix D"                << std::endl << D                 << std::endl << std::endl
           << "Matrix A+B"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix A+B"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix C+D"              << std::endl << A+B               << std::endl << std::endl
           << "Matrix E+3"              << std::endl << E+3               << std::endl << std::endl
           << "Matrix E*2"              << std::endl << E*2               << std::endl << std::endl
           << "Matrix E*A"              << std::endl << E*A               << std::endl << std::endl
           << "Matrix (C+1).*(E+3)"     << std::endl << (E+1).mul(E+3)    << std::endl << std::endl
           << "Matrix A^T" << std::endl << A.t()                          << std::endl << std::endl
  ;
  return 0;
}

int ImaGenesis(){
  cv::Mat BlackCat, BlancChat, BlancChatSize, LaRayita, LosHunos;
  BlackCat      = cv::Mat::zeros(128,256,CV_8UC1);
  BlancChat     = cv::Mat::ones (128,256,CV_8UC1);
  BlancChatSize = cv::Mat::ones (cv::Size(128,256),CV_8UC1); 
  LaRayita      = cv::Mat::eye  (200,200,CV_8UC1);
  LosHunos      = cv::Mat::eye  (200,200,CV_8UC1);

  cv::imshow("BlackCat",BlackCat);
  cv::imshow("BlancChat",BlancChat*255);
  cv::imshow("BlancChat cv::Size",BlancChatSize*255);
  cv::imshow("La Rayita xd",LaRayita*255);
  cv::imshow("La Rayita Gris xd",(LosHunos+LaRayita)*(255/2));

  cv::waitKey();
  return 0;
}

int ImaGenesisDos(){
  std::cout << "El regreso de cv::Mat";

  cv::Mat ImagenNueva(cv::Size(128,256),CV_8UC1);
  // Incremento Gradual de intensidad

  //ImagenNueva.rows=20;

  for(int i=0; i < ImagenNueva.rows ; ++i)
    for(int j=0; j < ImagenNueva.cols; ++j)
      ImagenNueva.at<uchar>(i,j) = j;

  cv::imshow("Degradado A",ImagenNueva);

  cv::waitKey();
  return 0;
}

int mainDatos(int argc, char** argv){
  if(argc<2) { std::cout << "Requires an argument (image)"; return 1; }
  std::cout << "Checking OpenCV installation in PC /opencv/ \n";
  cv::Mat InputOne, InputTwo, InputDri, InputFur;

  InputOne      = cv::imread(argv[1],cv::IMREAD_COLOR);
  cv::cvtColor(InputOne,InputTwo,cv::COLOR_BGR2GRAY);
  cv::cvtColor(InputTwo,InputDri,cv::COLOR_GRAY2BGR);

  if(InputOne.empty()) { std::cout << "Image reading error"; return 1; }
  cv::namedWindow("L'imageUn",cv::WINDOW_NORMAL);
  ImagData(InputOne);
  ImagData(InputTwo);
  ImagData(InputDri);

  cv::line(
      InputDri,
      cv::Point(0,0), cv::Point(20,50),
      cv::Scalar(0,255,0), 2, cv::LINE_8,0
  );
  cv::circle(
      InputDri,
      cv::Point(InputDri.cols/2,InputDri.rows/2),InputDri.rows/2,
      cv::Scalar(120,040,200),1, cv::LINE_8,0
  );

  cv::imshow("L'imageUn",InputOne);
  cv::imshow("L'imageDoux",InputTwo);
  cv::imshow("L'imageTrois",InputDri);

  cv::waitKey(0);
  return 0;
}

int ImagData(const cv::Mat& imagen){
  coutSEPARATOR
  std::cout << "The Image Has " << imagen.cols << " cols"
            << " And " << imagen.rows << " rows With Type"
  ;
  switch (imagen.type()){
    case CV_8UC1:
      std::cout << " 1 Channel | unsigned char\n";
      break;
    case CV_8UC3:
      std::cout << " 3 Channels | unsgned char\n";
      break;
    default:
      std::cout << " Undefined type\n";
      break;
  }
  coutSEPARATOR
  return 0;
}

int videoStreaming(void){
  cv::VideoCapture cam(0);
  if(!cam.isOpened()){ std::cout << "Camera Inaccessible!"; return 1; }

  // Image Capture Loop
  cv::Mat Frame;
  while(cam.read(Frame)){
    imshow("TheCam", Frame);
    if(cv::waitKey(30) > -1) break;
  }
  cv::destroyAllWindows();
  return 0;
}

int frameProcess(void){
  cv::VideoCapture cam(0);
  if(!cam.isOpened()){ std::cout << "Camera Inaccessible!"; return 1; }

  // Image Capture Loop
  cv::Mat Frame, Output;
  while(cam.read(Frame)){
    Frame.copyTo(Output);
    negateColour(Frame,Output);
    negateColour(Output);
    imshow("TheCam In",  Frame);
    imshow("TheCam Out", Output);
    if(cv::waitKey(30) > -1) break;
  }
  std::cout << "Resta uwu" << cv::sum(Frame-Output);
  cv::destroyAllWindows();
  return 0;
}

int negateColour(const cv::Mat& in, cv::Mat& out){
  out = cv::Scalar::all(255) - in;
  return 0;
}

int negateColour(cv::Mat& out){
  out = cv::Scalar::all(255) - out;
  return 0;
}

int frameROI(void){
  cv::VideoCapture cam(0);
  if(!cam.isOpened()){ std::cout << "Camera Inaccessible!"; return 1; }

  // Image Capture Loop
  cv::Mat Frame, Output, ROI;
  while(cam.read(Frame)){
    Frame.copyTo(Output);
    
    ROI = 
      Frame(
          cv::Range(0.25*Frame.rows, 0.75*Frame.rows),
          cv::Range(0.25*Frame.cols, 0.75*Frame.cols)
      );

    negateColour(ROI);

    imshow("TheCam In",  Frame);

    //ROI.copyTo(Output(cv::Rect(
    //        Output.cols/4 , Output.rows/4,
    //        Output.cols/2 , Output.rows/2
    //)));
    //
    ROI.copyTo(Output(cv::Rect(
            cv::Size(Output.cols/4 , Output.rows/4),
            cv::Size(Output.cols/2 , Output.rows/2)
    )));

    imshow("TheCam Out", Output);
    imshow("TheCam ROI", ROI);

    if(cv::waitKey(30) > -1) break;
  }
  cv::destroyAllWindows();
  return 0;
}

int mainHistograma(int argc, char** argv){
  if(argc < 2) {std::cout << "No file given" << std::endl; return 1;}

  std::cout << "Histograma" << std::endl;
  cv::Mat Input;
  Input = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
  if(Input.empty()) { std::cout << "Error File not Found" << std::endl; return 1; }

  cv::imshow("L'image gray",Input);
  lovdog_Histograma("L'image histograme",Input);

  cv::waitKey(0);
  return 0;
}

int lovdog_Histograma(const char* title, const cv::Mat& input){
  if(input.channels() != 1){ std::cout << "One Channel Images Only Here" << std::endl; return 1;}
  int histSize = 256;
  int histWidth = 512, histHeight = 200;
  cv::Scalar
    colorBack(20,0,0),
    colorBars(155,155,155),
    colorLines(255,0,255)
  ;

  float range[] = {0.0, (float)histSize};
  const float* histRange[] = { range };
  cv::Mat Histograma;

  cv::calcHist(&input, 1, 0, cv::Mat(), Histograma, 1, &histSize, histRange);
  cv::normalize(Histograma, Histograma, 0, histHeight, cv::NORM_MINMAX);
  std::cout << "Histogram Size" << Histograma.size();
  std::cout << " that is " << Histograma.rows << " rows, "
            << Histograma.cols << " columns. ";

  int binWidth = cvRound((double) histWidth/histSize);
  cv::Mat histogramaImg(histHeight, histWidth, CV_8UC3, colorBack);

  for( unsigned int i = 1 ; i < histSize ; ++i)
    line(
        histogramaImg,
        cv::Point(binWidth*(i), histHeight - cvRound(Histograma.at<float>(i)) ),
        cv::Point(binWidth*(i), histHeight ),
        colorBars, 2, 8, 0
    );

  for( unsigned int i = 1 ; i < histSize ; ++i)
    line(
        histogramaImg,
        cv::Point(binWidth*(i-1), histHeight - cvRound(Histograma.at<float>(i-1)) ),
        cv::Point(binWidth*(i), histHeight - cvRound(Histograma.at<float>(i)) ),
        colorLines, 2, 8, 0
    );

  imshow(title,histogramaImg);
  return 0;
}

int mainHistograma2(int argc, char** argv){
  if(argc < 2) {std::cout << "No file given" << std::endl; return 1;}

  std::cout << "Histograma" << std::endl;
  cv::Mat Input;
  Input = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
  if(Input.empty()) { std::cout << "Error File not Found" << std::endl; return 1; }

  cv::imshow("L'image gray",Input);
  lovdog_Histograma2("L'image histograme",Input);

  cv::waitKey(0);
  return 0;
}

int lovdog_Histograma2(const char* title, const cv::Mat& input){
  if(input.channels() != 1){ std::cout << "One Channel Images Only Here" << std::endl; return 1;}
  int histSize = 256;
  int histWidth = 512, histHeight = 200;
  cv::Scalar
    colorBack(20,0,0),
    colorBars(155,155,155),
    colorLines(255,0,255)
  ;

  float range[] = {0.0, (float)histSize};
  const float* histRange[] = { range };
  cv::Mat Histograma;

  if(!lovdogCalcHist(input, Histograma, 0, 0)){
    std::cout << "Histogram Error" << std::endl;
    return 1;
  }
  std::cout << Histograma << std::endl;
  //cv::calcHist(&i
  //cv::normalize(Histograma, Histograma, 0, histHeight, cv::NORM_MINMAX);
  //std::cout << "Histogram Size" << Histograma.size();
  //std::cout << " that is " << Histograma.rows << " rows, "
  //          << Histograma.cols << " columns. ";

  //int binWidth = cvRound((double) histWidth/histSize);
  //cv::Mat histogramaImg(histHeight, histWidth, CV_8UC3, colorBack);

  //for( unsigned int i = 1 ; i < histSize ; ++i)
  //  line(
  //      histogramaImg,
  //      cv::Point(binWidth*(i), histHeight - cvRound(Histograma.at<float>(i)) ),
  //      cv::Point(binWidth*(i), histHeight ),
  //      colorBars, 2, 8, 0
  //  );

  //for( unsigned int i = 1 ; i < histSize ; ++i)
  //  line(
  //      histogramaImg,
  //      cv::Point(binWidth*(i-1), histHeight - cvRound(Histograma.at<float>(i-1)) ),
  //      cv::Point(binWidth*(i), histHeight - cvRound(Histograma.at<float>(i)) ),
  //      colorLines, 2, 8, 0
  //  );

  //imshow(title,histogramaImg);
  return 0;
}

/* 
 * First argument is a Matrix which calculate the histogram from,
 * Second argument is the Matrix which will contain the histogram values
 * Third  argument, true for Probability Distribution, False for count
 * Fourth argument true for cumulative, false for standard
 * Return the bin number used by the histogram
 */
int lovdogCalcHist(const cv::Mat& src, cv::Mat &histogram1D, bool cnt_pdf, bool std_cum){
  if(src.empty()){
    std::cout << "\nlovdogCalcHist(): Empty Image" << std::endl;
    return 0;
  }
  if(src.channels() != 1){
    std::cout << "\nlovdogCalcHist(): Input must be 1 Channel" << std::endl;
    return 0;
  }
  if(histogram1D.empty())
    histogram1D = cv::Mat::zeros(1,HISTOGRAM_SIZE,CV_32FC1);
  switch(src.type()){
    case CV_8UC1:
      for(size_t theRow=0; theRow < src.rows ; ++theRow)
        for(size_t theCol=0; theCol<src.cols; ++theCol)
          ++histogram1D.at<float>(0, (int)src.at<uchar>(theRow,theCol) );
      break;

    case CV_32FC1:
      for(size_t theRow=0; theRow < src.rows ; ++theRow)
        for(size_t theCol=0; theCol<src.cols; ++theCol)
          ++histogram1D.at<float>(0, (int)src.at<float>(theRow,theCol) );
      break;

    default:
      std::cout << "\nlovdogCalcHist(): Unsupported Matrix Type" << std::endl;
  }

  // For Probability Distribution
  if(cnt_pdf) histogram1D/=(src.rows * src.cols);

  if(std_cum) for(int level=1; level < histogram1D.cols ; ++level)
    histogram1D.at<float>(0,level) += histogram1D.at<float>(0,level-1);

  return histogram1D.cols;
}

int lovdogShowHist(const cv::Mat& histogram, cv::Mat& ploutput, size_t width, size_t height){
  if(histogram.empty())         {std::cout<< "\nlovdogShowHist(): No Histogram to plot" <<std::endl; return 1 }
  if(histogram.channels() != 1) {std::cout<< "\nlovdogShowHist(): Histogram must be one channel matrix" <<std::endl; return 1;}
  if(histogram.type != CV_32FC1){std::cout<< "\nlovdogShowHist(): Histogram must be float type" <<std::endl; return 1;}
  if(ploutput.data)             {std::cout<< "\nlovdogShowHist(): Plot matrix must be empty" <<std::endl; return 1;}

  cv::Scalar
    ColorBackground =cv::Scalar(20,0,0),
    ColorBar        =cv::Scalar(160,10,150),
    ColorEdge       =cv::Scalar(200,0,200)
  ;

  int MaximumPosition = 0;
  float MaximumValue  = 0;
  for(size_t theCol=1; theCol<histogram.cols; ++theCol)
    if(histogram.at<float>(0,theCol) > MaximumValue)
      MaximumValue = histogram.at<float>(0,(MaximumPosition=theCol)); // Ahorro una línea, asignación y uso al mismo tiempo

  ploutput = cv::Mat::zeros(height,width,CV_8UC3);
  ploutput += ColorBackground;
  return 0;
}
