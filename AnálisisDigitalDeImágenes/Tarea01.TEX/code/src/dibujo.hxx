#ifndef __LOVDOG_DIBUJO__
#define __LOVDOG_DIBUJO__ value

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int GetEdge(const cv::Mat& lImage, cv::Mat& lImage_f, bool thres=true);
void showIt(cv::Mat& matrix, std::string name);
int lovdogEqualizeIt(const cv::Mat& src, cv::Mat& dst);

#endif /* ifndef __LOVDOG_DIBUJO__ */
