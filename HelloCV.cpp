#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main(void)
{
        Mat img = imread("lena.jpg");
 
        if (img.empty()) {
                cerr << "Image load failed!" << endl;
                return -1;
        }
 
        namedWindow("image");
        imshow("image", img);
        waitKey(0);
        return 0;
}