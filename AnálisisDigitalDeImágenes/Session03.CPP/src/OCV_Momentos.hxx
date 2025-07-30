#ifndef __LOVDOG_OCV_MOMENTOS__
#define __LOVDOG_OCV_MOMENTOS__
#include "OCV_Course.hxx"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

typedef std::vector<cv::Moments> MomentsVector;
typedef std::vector<cv::Point> Centros;

int mainMomentos(int argc, char** argv);
int lovdogMomentos(const Contornos& src, MomentsVector& mv);
int lovdogCentroides(const MomentsVector& mv, Centros& centroides);
int printCentroides(const Centros& centroides, const MomentsVector& mv, const Contornos& src);
int drawCentroid(const Centros& centroides, const Contornos& src, cv::Mat& dstImg);
#endif
