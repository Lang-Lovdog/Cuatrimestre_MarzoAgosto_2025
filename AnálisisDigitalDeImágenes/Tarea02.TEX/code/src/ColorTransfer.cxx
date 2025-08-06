#include "ColorTransfer.hxx"

namespace lovdog{
  float MatColProd(const float* Mat, size_t col, size_t numCols, size_t colSize, cv::Vec3f& op){
    float prod = 0;
    size_t y;
    y=0;
    while(y<colSize){
      prod += Mat[numCols*y+col] * op[y];
      y++;
    }
    return prod;
  }

  cv::Vec3d takahiro_BGR2lab(cv::Vec3d bgr) {
    cv::Vec3d LMS, lab;
    double L = 0.3811 * bgr[2] + 0.5783 * bgr[1] + 0.0402 * bgr[0];
    double M = 0.1967 * bgr[2] + 0.7244 * bgr[1] + 0.0782 * bgr[0];
    double S = 0.0241 * bgr[2] + 0.1288 * bgr[1] + 0.8444 * bgr[0];
    LMS[0] = L > eps ? std::log10(L) : std::log10(eps);
    LMS[1] = M > eps ? std::log10(M) : std::log10(eps);
    LMS[2] = S > eps ? std::log10(S) : std::log10(eps);

    lab[0] = 1 / std::sqrt(3) * (LMS[0] + LMS[1] + LMS[2]);
    lab[1] = 1 / std::sqrt(6) * (LMS[0] + LMS[1] - 2 * LMS[2]);
    lab[2] = 1 / std::sqrt(2) * (LMS[0] - LMS[1]);

    return lab;
  }

  cv::Vec3d takahiro_lab2BGR(cv::Vec3d lab) {
    cv::Vec3d LMS, bgr;
    double L = 1 / std::sqrt(3) * lab[0] + 1 / std::sqrt(6) * lab[1] + 1 / std::sqrt(2) * lab[2];
    double M = 1 / std::sqrt(3) * lab[0] + 1 / std::sqrt(6) * lab[1] - 1 / std::sqrt(2) * lab[2];
    double S = 1 / std::sqrt(3) * lab[0] - 2 / std::sqrt(6) * lab[1];

    LMS[0] = L > -5 ? std::pow(10, L) : eps;
    LMS[1] = M > -5 ? std::pow(10, M) : eps;
    LMS[2] = S > -5 ? std::pow(10, S) : eps;

    bgr[0] = 0.0497 * LMS[0] - 0.2439 * LMS[1] + 1.2045 * LMS[2];
    bgr[1] = -1.2186 * LMS[0] + 2.3809 * LMS[1] - 0.1624 * LMS[2];
    bgr[2] = 4.4679 * LMS[0] - 3.5873 * LMS[1] + 0.1193 * LMS[2];

    return bgr;
  }

  void takahiro_convert(const cv::Mat& input, cv::Mat& output, bool toLab){
    output = cv::Mat(input.size(), CV_64FC3);
    if(toLab){
      // BGR -> lab
      for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
          output.at<cv::Vec3d>(y, x) = takahiro_BGR2lab(input.at<cv::Vec3d>(y, x));
        }
      }
      return;
    }
    for (int y = 0; y < input.rows; ++y) {
      for (int x = 0; x < input.cols; ++x) {
        output.at<cv::Vec3d>(y, x) = takahiro_BGR2lab(input.at<cv::Vec3d>(y, x));
      }
    }
  }

  cv::Vec3f px_Matrix_BGR2LMS(const cv::Vec3f& op){
    return cv::Vec3f(
      _Matrix_RGB2LMS[0] * op[2] + _Matrix_RGB2LMS[1] * op[1] + _Matrix_RGB2LMS[2] * op[0],
      _Matrix_RGB2LMS[3] * op[2] + _Matrix_RGB2LMS[4] * op[1] + _Matrix_RGB2LMS[5] * op[0],
      _Matrix_RGB2LMS[6] * op[2] + _Matrix_RGB2LMS[7] * op[1] + _Matrix_RGB2LMS[8] * op[0]
    );
  }

  cv::Vec3f px_Matrix_LMS2BGR(const cv::Vec3f& op){
    return cv::Vec3f(
      op[0]*_Matrix_LMS2RGB[6] + op[1]*_Matrix_LMS2RGB[7] + op[2]*_Matrix_LMS2RGB[8],
      op[0]*_Matrix_LMS2RGB[3] + op[1]*_Matrix_LMS2RGB[4] + op[2]*_Matrix_LMS2RGB[5],
      op[0]*_Matrix_LMS2RGB[0] + op[1]*_Matrix_LMS2RGB[1] + op[2]*_Matrix_LMS2RGB[2] 
    );
  }

  cv::Vec3f px_Matrix_LMS2lAphaBeta(const cv::Vec3f& op){
    cv::Vec3f p = cv::Vec3f (
      op[0]>1e-5? std::log10(op[0]) : std::log(1e-5),
      op[1]>1e-5? std::log10(op[1]) : std::log(1e-5),
      op[2]>1e-5? std::log10(op[2]) : std::log(1e-5)
    );
    return cv::Vec3f(
      (p[0] + p[1] +   p[2]) * _Matrix_LMS2lAphaBeta[0],
      (p[0] + p[1] - 2*p[2]) * _Matrix_LMS2lAphaBeta[1],
      (p[0] - p[1]         ) * _Matrix_LMS2lAphaBeta[2]
    );
  }

  cv::Vec3f px_Matrix_BGR2lAphaBeta(const cv::Vec3f& op){
    cv::Vec3f LMS = cv::Vec3f(
      _Matrix_RGB2LMS[0] * op[2] + _Matrix_RGB2LMS[1] * op[1] + _Matrix_RGB2LMS[2] * op[0],
      _Matrix_RGB2LMS[3] * op[2] + _Matrix_RGB2LMS[4] * op[1] + _Matrix_RGB2LMS[5] * op[0],
      _Matrix_RGB2LMS[6] * op[2] + _Matrix_RGB2LMS[7] * op[1] + _Matrix_RGB2LMS[8] * op[0]
    );
    LMS = cv::Vec3f(
      LMS[0] > eps ? std::log10(LMS[0]) : std::log10(eps),
      LMS[1] > eps ? std::log10(LMS[1]) : std::log10(eps),
      LMS[2] > eps ? std::log10(LMS[2]) : std::log10(eps)
    );
    return cv::Vec3f(
      _Matrix_LMS2lAphaBeta[0] * (LMS[0] + LMS[1] + LMS[2]),
      _Matrix_LMS2lAphaBeta[1] * (LMS[0] + LMS[1] - 2 * LMS[2]),
      _Matrix_LMS2lAphaBeta[2] * (LMS[0] - LMS[1])
    );
  }
  cv::Vec3f px_Matrix_lAphaBeta2LMS(const cv::Vec3f& op){
    cv::Vec3f p = cv::Vec3f (
      op[0]>-5?std::pow(op[0],10):1e-5,
      op[1]>-5?std::pow(op[1],10):1e-5,
      op[2]>-5?std::pow(op[2],10):1e-5
    );
    return cv::Vec3f(
      p[0]*_Matrix_lAphaBeta2LMS[0] + p[1]*_Matrix_lAphaBeta2LMS[1] + p[2]*_Matrix_lAphaBeta2LMS[2],
      p[0]*_Matrix_lAphaBeta2LMS[0] + p[1]*_Matrix_lAphaBeta2LMS[1] - p[2]*_Matrix_lAphaBeta2LMS[2],
      p[0]*_Matrix_lAphaBeta2LMS[0] - p[1]*_Matrix_lAphaBeta2LMS[1]
    );
  }
  cv::Vec3f px_Matrix_lAphaBeta2BGR(const cv::Vec3f& op){
    cv::Vec3f LMS = cv::Vec3f(
      _Matrix_LMS2lAphaBeta[0]*op[0] +   _Matrix_LMS2lAphaBeta[1]*op[1] + _Matrix_LMS2lAphaBeta[2]*op[2],
      _Matrix_LMS2lAphaBeta[0]*op[0] +   _Matrix_LMS2lAphaBeta[1]*op[1] - _Matrix_LMS2lAphaBeta[2]*op[2],
      _Matrix_LMS2lAphaBeta[0]*op[0] - 2*_Matrix_LMS2lAphaBeta[1]*op[1]
    );
    LMS[0] = LMS[0] > -5 ? std::pow(10, LMS[0]) : eps;
    LMS[1] = LMS[1] > -5 ? std::pow(10, LMS[1]) : eps;
    LMS[2] = LMS[2] > -5 ? std::pow(10, LMS[2]) : eps;
    return cv::Vec3f(
      _Matrix_LMS2RGB[6]*LMS[0] + _Matrix_LMS2RGB[7]*LMS[1] + _Matrix_LMS2RGB[8]*LMS[2],
      _Matrix_LMS2RGB[3]*LMS[0] + _Matrix_LMS2RGB[4]*LMS[1] + _Matrix_LMS2RGB[5]*LMS[2],
      _Matrix_LMS2RGB[0]*LMS[0] + _Matrix_LMS2RGB[1]*LMS[1] + _Matrix_LMS2RGB[2]*LMS[2]
    );
  }

  void cvtColor(const cv::Mat& input, cv::Mat& output, uint code){
    if(input.empty()) return;

    cv::Mat inputF;
    cv::Vec3f op;
    size_t i, j;

    input.convertTo(inputF, CV_32FC3, 1/255.0);

    output = cv::Mat::zeros(input.size(), CV_32FC3);


    switch(code){
      case BGR2LMS:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j) = px_Matrix_BGR2LMS(inputF.at<cv::Vec3f>(i,j));
           ++j; 
          }
          ++i;
        }
        break;

      case LMS2BGR:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j) = px_Matrix_LMS2BGR(inputF.at<cv::Vec3f>(i,j));
            ++j;
          }
          ++i;
        }
        break;

      case LMS2lAlphaBeta:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j) = px_Matrix_LMS2lAphaBeta(inputF.at<cv::Vec3f>(i,j));
            ++j;
          }
          ++i;
        }
        break;

      case lAlphaBeta2LMS:
        if(input.channels() != 3) return;
        i=0; while( i < output.rows ){
          j=0; while( j < output.cols ){
            output.at<cv::Vec3f>(i,j) = 
              px_Matrix_lAphaBeta2BGR(inputF.at<cv::Vec3f>(i,j));
            ++j;
          }
          ++i;
        }
        break;

      case BGR2lAlphaBeta:
        i=0; while(i<output.rows){
          j=0; while(j<output.cols){
            output.at<cv::Vec3f>(i,j) =
              px_Matrix_BGR2lAphaBeta(inputF.at<cv::Vec3f>(i,j)); ++j;
          } ++i;
        }

      default:
        return;
    }
  }

  void colorTransfer(const cv::Mat& input, const cv::Mat& reference, cv::Mat& output){
    cv::Mat lABMsrc, lABMdst, LMSout, LMS;
    cv::Scalar meanSrc, meanDst;
    cv::Scalar stdSrc, stdDst;
    cv::Vec3f srcPX, dstPX;
    //size_t i, j;

    takahiro_convert(input, LMS, false);
    /*
    cvtColor(LMS, lABMsrc, LMS2lAlphaBeta);


    cvtColor(reference, LMS, BGR2LMS);
    cvtColor(LMS, lABMdst, LMS2lAlphaBeta);

    // Getting mean for each channel
    // Get std dev for each channel
    cv::meanStdDev(lABMsrc,meanSrc,stdSrc);
    cv::meanStdDev(lABMdst,meanDst,stdDst);


    //i=0; while(i<lABMsrc.rows){
    //  j=0; while(j<lABMsrc.cols){
    //    srcPX = lABMsrc.at<cv::Vec3f>(i,j);
    //    dstPX[0] = (srcPX[0] - meanSrc[0])*stdDst[0]/stdSrc[0] + meanDst[0];
    //    dstPX[1] = (srcPX[1] - meanSrc[1])*stdDst[1]/stdSrc[1] + meanDst[1];
    //    dstPX[2] = (srcPX[2] - meanSrc[2])*stdDst[2]/stdSrc[2] + meanDst[2];
    //    lABMsrc.at<cv::Vec3f>(i,j) = dstPX;
    //    ++j;
    //  }
    //  ++i;
    //}

    cvtColor(lABMsrc, LMS, lAlphaBeta2LMS);
    */
    takahiro_convert(LMS, output, true);
//    output.convertTo(output, CV_8UC3, 255.0);
    showIt(output, "Output", false);
  }

  int mainColorTransfer(int argc, char** argv){
    cv::Mat input, reference, output;
    std::string src_arg = argv[1];
    std::string ref_arg = argv[2];
    //bool outusr = (argc > 3);
    lovdogGetImage(src_arg, input);
    showIt(input, "Input", false);
    lovdogGetImage(argv[2], reference);
    showIt(reference, "Reference", false);
    colorTransfer(input, reference, output);
    //showIt(output, outusr ? std::string(argv[3]) : "Output" , outusr);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
  }
};
