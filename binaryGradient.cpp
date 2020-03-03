#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;

const int imageNo = 6;
 
int main(int argc, char* argv[])
{

    IplImage *image[imageNo];
    IplImage *HSV[imageNo];
    vector<Mat> hsv_planes[imageNo];
    char filename[1024];
    for(int index = 0; index < imageNo; index++)
    {
        sprintf(filename, "%d", index);
        image[index] = cvLoadImage(filename);
        cvCvtColor(image[index], HSV[index], CV_BGR2HSV);

        split(HSV[index], hsv_planes[index]);
    }





    
        namedWindow("image");
        //imshow("image", img);
        waitKey(0);
        return 0;
}