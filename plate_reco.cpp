#include "opencv/cv.hpp"
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <math.h>
 
 using namespace std;
 using namespace cv;
 
int main( int argc, char** argv )
{

//declaring matrices for storing images
Mat src; 

//declaring pixel vector to store hsv values of image

// src stores input image
src=imread(argv[1]);
Mat org1;
Mat org;
src.copyTo(org1);
cvtColor(src,src,CV_BGR2GRAY);
src.copyTo(org);
Mat kernel1 = Mat::ones(Size(5, 11), CV_8U);
Mat kernel = Mat::ones(Size(3, 9), CV_8U);
dilate(src,src, kernel);
imshow("dilated",src);

erode(src,src,kernel);
imshow("eroded",src);

imshow("input",org);

Mat im1;
Mat im2,output;
src.copyTo(im1);
org.copyTo(im2);
Mat output2 = im1-im2;
imshow("subtraction",output2);

GaussianBlur(output2, output2,Size(5,5), 0, 0);
Canny(output2,output,128, 255,3);
imshow("canny_output",output);

//finding the histogram thing
int values_y[output.rows];
int values_x[output.cols];

for(int i=0; i<output.rows;i++)
{
    values_y[i]=0;
    for (int j= 0; j<output.cols; j++)
    {
        values_y[i]+=output.at<uchar>(j,i);
    }
}
for(int i=0; i<output.cols;i++)
{
    values_x[i]=0;
    for (int j= 0; j<output.rows; j++)
    {
        values_x[i]+=output.at<uchar>(j,i); 
    }
}
Mat new_img;
output.copyTo(new_img);
int threshold_x=10*output.rows;
int threshold_y=30*output.cols;
for(int i=0; i<output.rows;i++) {
    if(values_x[i]<threshold_x) {
        for(int j=0; j<output.cols; ++j) {
            new_img.at<uchar>(i,j)=0;
        }
    }
}
 

for(int i=0; i<output.cols;i++) {
    if(values_y[i]<threshold_y) {
        for(int j=0; j<output.rows; ++j) {
            new_img.at<uchar>(j,i)=0;
        }
    }
}
imshow("new_img",new_img);     

int matrices[output.cols][output.rows];
// long long avg_val=0;
// for(int i=0;i<output.cols;i++)
// {
//     for(int j=0;j<output.rows;j++)
//     {
//         matrices[i][j]=(values_x[i])&(values_y[j]);
//         avg_val+=matrices[i][j];
//     }
// }

// avg_val=avg_val/((output.cols)*(output.rows));
// cout<<"avg_val="<<avg_val;
//    for(int i=0; i<output.rows;i++)
// {
//     for (int j= 0; j<output.cols; j++)
//     {
//         if(matrices[i][j]>2*avg_val) {

//            // cout<<" "<<matrices[i][j];
//         }
//     }
//  //   cout<<"\n";
// }


dilate(new_img,new_img, kernel);
//erode(new_img,new_img,kernel);
imshow("imm",new_img);


//Mat result;
//cv::threshold(new_img, result, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
//imshow("threshold output",result);

  std::vector<std::vector<cv::Point> > contours; 
findContours(new_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); 

vector <vector<Point> >contours_poly(contours.size());
vector<Rect> boundRect(contours.size());
vector<Point2f> center(contours.size());
vector<float>radius(contours.size());

//  for (size_t i = 0; i < contours.size(); i++) {
// approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
// boundRect[i]=boundingRect(Mat( contours_poly[i]));
// minEnclosingCircle((matrices)contours_poly[i],center[i],radius[i]);
// }

cv::Mat contourImage(new_img.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
/*int max_area=0;
int max_area_idx=0;
int average_area=0;
int sum_area=0;
 for (size_t idx = 0; idx < contours.size(); idx++) {
if(contourArea(contours[idx])>=max_area)
{
max_area_idx=idx;
max_area=contourArea(contours[idx]);
sum_area+=contourArea(contours[idx]);
}
}
average_area=sum_area/(contours.size());*/
//cout<<average_area;
//cout<<max_area;
 for (size_t idx = 0; idx < contours.size(); idx++) {
//if(contourArea(contours[idx])>=average_area)
      //  cv::drawContours(contourImage, contours, idx, colors[0]);
           cv::drawContours(contourImage, contours, idx, colors[2]);

rectangle(contourImage,boundRect[idx].tl(),boundRect[idx].br(), colors[1],2,8,0);
}
imshow("contour",contourImage);
    //std::vector rects;  
/*
       std::vector<std::vector >::iterator itc = contours.begin();  
        while (itc != contours.end())  
        {  
        cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*itc));     
              
            float area = fabs(cv::contourArea(*itc));  
                float bbArea=mr.size.width * mr.size.height;  
                float ratio = area/bbArea;  
                      
                if( (ratio < 0.45) || (bbArea < 400) ){  
                    itc= contours.erase(itc);  
                }else{  
                    ++itc;  
                    rects.push_back(mr);  
                }  
        }  


/*

for(int i=0; i<result.cols; i++)
{
for(int j=0;j<result.rows;j++)
{ 
if(result.at<uchar>(j,i)==255)
{new_img.at<Vec3b>(j,i)[0]=255;
new_img.at<Vec3b>(j,i)[1]=0;
new_img.at<Vec3b>(j,i)[2]=0;
} 
else
{
new_img.at<Vec3b>(j,i)[0]=0;
new_img.at<Vec3b>(j,i)[1]=0;
new_img.at<Vec3b>(j,i)[2]=0;
}}}
imshow("new_img",new_img);
 std::vector<std::vector<cv::Point> > rects;  
    std::vector rects;  
       std::vector<std::vector >::iterator itc = contours.begin();  
        while (itc != contours.end())  
        {  
        cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*itc));     
              
            float area = fabs(cv::contourArea(*itc));  
                float bbArea=mr.size.width * mr.size.height;  
                float ratio = area/bbArea;  
                      
                if( (ratio < 0.45) || (bbArea < 400) ){  
                    itc= contours.erase(itc);  
                }else{  
                    ++itc;  
                    rects.push_back(mr);  
                }  
        }  
   
    cv::threshold(img, img_bin, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);  

    std::vector< std::vector > contours;  
    cv::findContours(img_bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);  

    cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*maxAreaContour));  
*/
//imshow("imm1",result);

//result=result+src;
//std::vector< std::vector > contours;  
//cv::findContours(result, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//imshow("imm1",output);
//Sobel(output, output, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
//imshow("output",output);
//imshow("result",result);
cvWaitKey();
return 0;
}
