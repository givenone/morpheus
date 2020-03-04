#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;

const int imageNo = 6;
int width, height;

void albedo(IplImage**& binary_images, CvMat**& mixed, CvMat*& diffuse, CvMat*& specular)
{


    for(int i=0; i<imageNo/2; i++)
    {
        mixed[i] = cvCreateMat(width, height, CV_8UC3);
        cvAdd(binary_images[2*i], binary_images[2*i + 1], mixed[i]);
    }

    IplImage **HSV = new IplImage* [imageNo];

    for(int index = 0; index < imageNo; index++)
    {
        cvCvtColor(binary_images[index], HSV[index], CV_BGR2HSV);
    }

    CvMat *delta[3];
    CvMat *Chroma[2];
    Chroma[0] = cvCreateMat(width, height, CV_8UC1);
    Chroma[1] = cvCreateMat(width, height, CV_8UC1);

    for(int index = 0; index < imageNo/2; index++)
    {
        CvMat* Hue = cvCreateMat(width,height, CV_8UC1);
        CvMat* Intensity = cvCreateMat(width,height, CV_8UC1);
        CvMat* Saturation = cvCreateMat(width,height, CV_8UC1);

        CvMat* Hue_c = cvCreateMat(width,height, CV_8UC1);
        CvMat* Intensity_c = cvCreateMat(width,height, CV_8UC1);
        CvMat* Saturation_c = cvCreateMat(width,height, CV_8UC1);

        cvSplit(HSV[index*2], Hue, Saturation, Intensity, NULL );
        cvSplit(HSV[index*2+1], Hue_c, Saturation_c, Intensity, NULL);

        delta[index] = cvCreateMat(width,height, CV_8UC1);
        
        for(int i=0; i<width; i++)
        {
            for(int j=0; j<height; j++)
            {
                char* rgb_1 = binary_images[index*2]->imageData + binary_images[index*2]->widthStep * j + i * 3;
                Chroma[0]->data.i[i*width + j] = max(rgb_1[0], max(rgb_1[1], rgb_1[2])) - min(rgb_1[0], min(rgb_1[1], rgb_1[2]));
                char* rgb_2 = binary_images[index*2+1]->imageData + binary_images[index*2+1]->widthStep * j + i * 3;
                Chroma[1]->data.i[i*width + j] = max(rgb_1[0], max(rgb_1[1], rgb_1[2])) - min(rgb_1[0], min(rgb_1[1], rgb_1[2]));

                delta[index]->data.i[i*width + j] = (Intensity->data.i[i*width + j] > Intensity_c->data.i[i*width + j]) ?
                Intensity->data.i[i*width + j] - Chroma[0]->data.i[i*width + j] / Saturation_c->data.i[i*width + j] :
                Intensity_c->data.i[i*width + j] - Chroma[1]->data.i[i*width + j] / Saturation->data.i[i*width + j];
            }
        }
    }

    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
            int *specular_data = (int*)cvPtr2D(specular, i, j);
            int *diffuse_data = (int*)cvPtr2D(diffuse, i, j);
            int *mixed_data = (int*)cvPtr2D(mixed, i, j);
            specular_data[0] = delta[0]->data.i[i*width + j];
            specular_data[1] = delta[0]->data.i[i*width + j];
            specular_data[2] = delta[0]->data.i[i*width + j];
            diffuse_data[0] =  mixed_data[0] - specular_data[0];
            diffuse_data[1] =  mixed_data[1] - specular_data[1];
            diffuse_data[2] =  mixed_data[2] - specular_data[2]; 
        }
    }


}

void normal()
{}

int main(int argc, char* argv[])
{

    IplImage **image;
    image = new IplImage* [imageNo];
    char filename[1024];
    for(int index = 0; index < imageNo; index++)
    {
        sprintf(filename, "%d", index);
        image[index] = cvLoadImage(filename);
    }
    int width = image[0]->width; 
    int height = image[0]->height;


    CvMat** mixed_albedo;
    CvMat *specular_albedo, *diffuse_albedo;
    mixed_albedo = new CvMat* [imageNo / 2];
    mixed_albedo[0] = cvCreateMat(width, height, CV_8UC3);
    mixed_albedo[1] = cvCreateMat(width, height, CV_8UC3);
    mixed_albedo[2] = cvCreateMat(width, height, CV_8UC3);
    specular_albedo = cvCreateMat(width, height, CV_8UC3);
    diffuse_albedo = cvCreateMat(width, height, CV_8UC3);

    albedo(image, mixed_albedo, diffuse_albedo, specular_albedo);

    namedWindow("image");
    cvShowImage("image", specular_albedo);

    waitKey(0);
    return 0;
}