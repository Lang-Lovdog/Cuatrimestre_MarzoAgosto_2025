#include "ColorTransfer.hxx"

namespace lovdog{
  float MatColProd(const float* Mat, size_t col, size_t numCols, size_t colSize, float op){
    float prod = 0;
    size_t y;
    y=0;
    while(y<colSize){
      prod += Mat[numCols*y+col] * op;
      y++;
    }
    return prod;
  }

  void cvtColor(const cv::Mat& input, cv::Mat& output, uint code){
    if(input.empty()) return;
    //if(!output.empty()) return;

    //cv::Mat output;
    cv::Mat inputF;
    size_t i, j;

    input.convertTo(inputF, CV_32FC3);

    output = cv::Mat::zeros(input.rows, input.cols, CV_32FC3);


    switch(code){
      case BGR2LMS:
        if(input.channels() != 3) return;
        std::cout << "BGR2LMS" << std::endl;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j)[0] = std::log( MatColProd(_Matrix_RGB2LMS, 0, 3, 3, inputF.at<cv::Vec3f>(i,j)[2]) + 1e-8 );
            output.at<cv::Vec3f>(i,j)[1] = std::log( MatColProd(_Matrix_RGB2LMS, 1, 3, 3, inputF.at<cv::Vec3f>(i,j)[1]) + 1e-8 );
            output.at<cv::Vec3f>(i,j)[2] = std::log( MatColProd(_Matrix_RGB2LMS, 2, 3, 3, inputF.at<cv::Vec3f>(i,j)[0]) + 1e-8 );
            ++j;
          }
          ++i;
        }
        break;

      case LMS2BGR:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j)[0] = MatColProd(_Matrix_LMS2RGB, 0, 3, 3, inputF.at<cv::Vec3f>(i,j)[2]) + 1e-8;
            output.at<cv::Vec3f>(i,j)[1] = MatColProd(_Matrix_LMS2RGB, 1, 3, 3, inputF.at<cv::Vec3f>(i,j)[1]) + 1e-8;
            output.at<cv::Vec3f>(i,j)[2] = MatColProd(_Matrix_LMS2RGB, 2, 3, 3, inputF.at<cv::Vec3f>(i,j)[0]) + 1e-8;
            ++j;
          }
          ++i;
        }
        break;

      case LMS2lAlphaBeta:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j)[0] = (inputF.data[i] + inputF.data[i+1] +   inputF.data[i+2]) * _Matrix_LMS2lAphaBeta[0];
            output.at<cv::Vec3f>(i,j)[1] = (inputF.data[i] + inputF.data[i+1] - 2*inputF.data[i+2]) * _Matrix_LMS2lAphaBeta[1];
            output.at<cv::Vec3f>(i,j)[2] = (inputF.data[i] - inputF.data[i+1]                     ) * _Matrix_LMS2lAphaBeta[2];
            ++j;
          }
          ++i;
        }
        break;

      case lAlphaBeta2LMS:
        float l3, a6, b2;
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){

            l3 = inputF.at<cv::Vec3f>(i,j)[0]*_Matrix_LMS2lAphaBeta[0]/3;
            a6 = inputF.at<cv::Vec3f>(i,j)[1]*_Matrix_LMS2lAphaBeta[1]/6;
            b2 = inputF.at<cv::Vec3f>(i,j)[2]*_Matrix_LMS2lAphaBeta[2]/2;

            output.at<cv::Vec3f>(i,j)[0] = std::exp(l3 + a6 +   b2);
            output.at<cv::Vec3f>(i,j)[1] = std::exp(l3 + a6 - 2*b2);
            output.at<cv::Vec3f>(i,j)[2] = std::exp(l3 - a6       );

            ++j;
          }
          ++i;
        }
        break;

      default:
        return;
    }
  }

  void colorTransfer(const cv::Mat& input, const cv::Mat& reference, cv::Mat& output){
    cv::Mat lABMsrc, lABMdst, LMS;
    cv::Scalar meanSrc, meanDst;
    cv::Scalar stdSrc, stdDst;
    Gallery srcChannels(3), dstChannels(3);

    cvtColor(input, LMS, BGR2LMS);
    std::cout << LMS << std::endl;
    cvtColor(LMS, lABMsrc, LMS2lAlphaBeta);


    cvtColor(reference, LMS, BGR2LMS);
    cvtColor(LMS, lABMdst, LMS2lAlphaBeta);

    // Getting mean for each channel
    meanSrc = cv::mean(lABMsrc);
    meanDst = cv::mean(lABMdst);

    // Get std dev for each channel
    stdSrc = cv::mean((lABMsrc - meanSrc).mul(lABMsrc - meanSrc));
    stdDst = cv::mean((lABMdst - meanDst).mul(lABMdst - meanDst));

    cv::split(lABMsrc, srcChannels);
    cv::split(lABMdst, dstChannels);

    srcChannels[0] = (srcChannels[0]-meanSrc[0])*stdDst[0]/stdSrc[0] + meanDst[0];
    srcChannels[1] = (srcChannels[1]-meanSrc[1])*stdDst[1]/stdSrc[1] + meanDst[1];
    srcChannels[2] = (srcChannels[2]-meanSrc[2])*stdDst[2]/stdSrc[2] + meanDst[2];

    cv::merge(srcChannels, LMS);
    cvtColor(LMS, output, lAlphaBeta2LMS);
    cvtColor(output, output, LMS2BGR);
  }

  int mainColorTransfer(int argc, char** argv){
    cv::Mat input, reference, output;
    std::string src_arg = argv[1];
    std::string ref_arg = argv[2];
    bool outusr = (argc > 3);
    lovdogGetImage(src_arg, input);
    showIt(input, "Input", false);
    lovdogGetImage(argv[2], reference);
    showIt(reference, "Reference", false);
    colorTransfer(input, reference, output);
    showIt(output, outusr ? std::string(argv[3]) : "Output" , outusr);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
  }
};
