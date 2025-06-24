#include "MatrizCaracterísticas.hxx"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace lovdog{
  void crop_from_mask(
      cv::Mat &src,
      cv::Mat &mask,
      cv::Mat **dst,
      size_t radius
  ){
    if( (src.dims | mask.dims) > 1){
      std::cerr << "Images must be on gray scale" << std::endl
                << "Converting..." << std::endl;
    }
    *dst = new cv::Mat[5];
    // Here the program will search the white pixel
    // and position the anchor point, if
    // anchor = 0, it will be the top left corner
    // anchor = 1, it will be the top right corner
    // anchor = 2, it will be the bottom left corner
    // anchor = 3, it will be the bottom right corner
    // anchor = 4, it will be the center of the image
    int topLeft[2] = { 0, 0 };
    int topRight[2] = { 0, 0 };
    int bottomLeft[2] = { 0, 0 };
    int bottomRight[2] = { 0, 0 };
    int center[2] = { 0, 0 };
    int anchor_x = 0;
    int anchor_y = 0;
    int i,j;
    bool break_flag = false;
    
    // Here the program will iterate over the mask
    // searching the top left corner (white pixel mask)
    i=j=0;
    while (i < mask.rows && break_flag){ j=0;
      while (j < mask.cols && break_flag){
        if (mask.at<uchar>(i, j) == 255) {
          topLeft[0] = j;
          topLeft[1] = i;
          break_flag = true;
        } ++j;
      } ++i;
    }
    break_flag = false;
    // Here the program will iterate over the mask
    // searching the top right corner (white pixel mask)
    i=j=0;
    while(i < mask.rows && break_flag){ j=mask.cols;
      while(j > 0 && break_flag){ --j;
        if (mask.at<uchar>(i, j) == 255) {
          topRight[0] = j;
          topRight[1] = i;
          break_flag = true;
        }
      } ++i;
    }
    break_flag = false;
    // Here the program will iterate over the mask
    // searching the bottom left corner (white pixel mask)
    i=mask.rows;
    while(i > 0 && break_flag){ --i; j=0;
      while(j < mask.cols && break_flag){
        if (mask.at<uchar>(i, j) == 255) {
          bottomLeft[0] = j;
          bottomLeft[1] = i;
          break_flag = true;
        } ++j;
      }
    }
    break_flag = false;
    // Here the program will iterate over the mask
    // searching the bottom right corner (white pixel mask)
    i=mask.rows;
    while(i > 0){ --i; j=mask.cols;
      while(j > 0){ --j;
        if (mask.at<uchar>(i, j) == 255) {
          bottomRight[0] = j;
          bottomRight[1] = i;
          break;
        }
      }
    }
    // The central anchor should be calculated by the
    // four points diagonals intersection
    // to say topleft - bottomright intersection with 
    // topright - bottomleft. Using the line equation
    // y = mx + b
    // m = (bottomleft[1] - topright[1]) / (bottomleft[0] - topright[0])
    // b = bottomleft[1] - m * bottomleft[0]
    // y = m * x + b
    // x = (y - b) / m
    depseek::find_intersection(
        topLeft, bottomRight,
        topRight, bottomLeft,
        center,(center+1)
    );
    // Here the program will crop the image
    // The crop will have the centers obtained before
    // and will have a sqared size of 2 * radius
    
    // First crop of radius x radius will be centered
    // at the topLeft corner
    **dst = 
      src(cv::Range(topLeft[1] - radius, topLeft[1] + radius),
          cv::Range(topLeft[0] - radius, topLeft[0] + radius));
    // Then crop of radius x radius will be centered
    // at the topRight corner
    *(*dst+1) = 
      src(cv::Range(topRight[1] - radius, topRight[1] + radius),
          cv::Range(topRight[0] - radius, topRight[0] + radius));
    // Then crop of radius x radius will be centered
    // at the bottomLeft corner
    *(*dst+2) = 
      src(cv::Range(bottomLeft[1] - radius, bottomLeft[1] + radius),
          cv::Range(bottomLeft[0] - radius, bottomLeft[0] + radius));
    // Then crop of radius x radius will be centered
    // at the bottomRight corner
    *(*dst+3) = 
      src(cv::Range(bottomRight[1] - radius, bottomRight[1] + radius),
          cv::Range(bottomRight[0] - radius, bottomRight[0] + radius));
    // And the last one will be the central crop
    *(*dst+4) = 
      src(cv::Range(center[1] - radius, center[1] + radius),
          cv::Range(center[0] - radius, center[0] + radius));
  }

};

namespace depseek {
  int find_intersection(const int p1[2], const int p2[2],
                       const int p3[2], const int p4[2],
                       int* ix, int* iy) {
      const int x1 = p1[0], y1 = p1[1];
      const int x2 = p2[0], y2 = p2[1];
      const int x3 = p3[0], y3 = p3[1];
      const int x4 = p4[0], y4 = p4[1];

      const int denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
      if (denom == 0.0) return 0;  // Lines are parallel

      const int t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
      const int s = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

      if (t >= 0.0 && t <= 1.0 && s >= 0.0 && s <= 1.0) {
          *ix = x1 + t * (x2 - x1);
          *iy = y1 + t * (y2 - y1);
          return 1;
      }

      return 0;  // Segments don't intersect
  }
};
