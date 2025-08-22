#ifndef __LOVDOG__DETECT_HXX__
#define __LOVDOG__DETECT_HXX__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

int mainDetect(int argc, char** argv);
int mainDetectCirc(int argc, char** argv);
int lovdogDibujaLineas(std::vector<cv::Vec4i> linesP, cv::Mat &lImageC);
int lovdogDibujaCirculos(std::vector<cv::Vec3f> linesP, cv::Mat &lImageC);

#endif
