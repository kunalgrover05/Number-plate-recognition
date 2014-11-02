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
Mat org;
src.copyTo(org);
//declaring pixel vector to store hsv values of image

// src stores input image
src=imread(argv[1]);
Mat boss;
src.copyTo(boss);
imshow("captured",src);

//resize(src,src,cvSize(src.cols/5,src.rows/5));
//imshow("resized",src);
/*Mat hsv;
cvtColor(src,hsv,CV_BGR2HSV);
//Vec3b pix;

imshow("hsv_before",hsv);
for(int i=0;i<hsv.rows;i++)
{
	for(int j=0;j<hsv.cols;j++)
	{
		if(hsv.at<Vec3b>(i,j)[2] >90)
			hsv.at<Vec3b>(i,j)[2]=60;
	}
}
imshow("hsv_after",hsv);
cvtColor(hsv,src,CV_HSV2BGR);
imshow("without illumination",src);
*/
cvtColor(src,src,CV_BGR2GRAY);
imshow("grayscaled",src);
/*
Mat kernel = Mat::ones(Size(11, 11), CV_8U);
Mat kernel1 = Mat::ones(Size(11, 11), CV_8U);
/*Mat kernel = Mat::ones(Size(3, 9), CV_8U);

dilate(src,src, kernel);
//imshow("dilated",src);

erode(src,src,kernel);
//imshow("eroded",src);

Mat im1;
Mat im2,output;
src.copyTo(im1);
org.copyTo(im2);
Mat output2 = im1-im2;
//imshow("subtraction",output2);*/

//output2.copyTo(src);


//USE TRESSECA- On all blobs
// Now aspect ratio for numbers
// Mode for areas
// Adaptive thresholding

//blur(src,src,Size(5,5));//0,0);

GaussianBlur(src,src,Size(5,5),0,0);///using this instead of median for binarization
//GaussianBlur(src,src,Size(3,3),0,0);
imshow("blurred",src);
/*
Mat er1;
Mat dil1;
Mat open;
Mat closed;
erode(src,er1,kernel1);
dilate(er1,open,kernel1);
imshow("open",open);

dilate(src,dil1,kernel1);
erode(dil1,closed,kernel1);
imshow("closed",closed);

src=closed-open;
imshow("subtraction",src);

dilate(src,src,kernel);
erode(src,src,kernel);
erode(src,src,kernel);
dilate(src,src,kernel);
imshow("diff_after",src);
//Mat frame;
//cv::GaussianBlur(src, frame, cv::Size(0, 0), 3);
//cv::addWeighted(frame, -0.7, src, 1.7, 0, src);
//imshow("sharpened",src);
//adaptiveThreshold( src, src, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3, 1 );
//adaptiveThreshold(src, src, , CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 0);
//imshow("adapt threshold",src);
/*Mat kernel = Mat::ones(Size(3, 3), CV_8U);

dilate(src,src, kernel);
imshow("dilated",src);

erode(src,src,kernel);
imshow("eroded",src);*/


cv::threshold(src, src, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);	//not using adaptive thresholding etc//use inv-imp

//cv::threshold(src, src, 5, 255, CV_THRESH_BINARY);//_INV|CV_THRESH_OTSU);	//not using adaptive thresholding etc//use inv-imp

imshow("threshold",src);

Mat binary=src;
vector < vector<cv::Point>  > blobs;
            blobs.clear();

            // Fill the label_image with the blobs
            // 0  - background
            // 1  - unlabelled foreground
            // 2+ - labelled foreground

            ///input is a binary image therefore values are either 0 or 1
            ///out objective is to find a set of 1's that are together and assign 2 to it
            ///then look for other 1's, and assign 3 to it....so on a soforth

            cv::Mat label_image;
            binary.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S! Because the CV::SCALAR is a double value in the function floodfill

            int label_count = 2; // starts at 2 because 0,1 are used already

            //erode to remove noise-------------------------------
            Mat element = getStructuringElement( MORPH_RECT,
            Size( 2*3 + 1, 2*3+1 ),
            Point( 0, 0 ) );
            /// Apply the erosion operation
            //erode( label_image, label_image, element );
            //---------------------------------------------------

            //just check the Matrix of label_image to make sure we have 0 and 1 only
            //cout << label_image << endl;

            for(int y=0; y < binary.rows; y++) {
                for(int x=0; x < binary.cols; x++) {
                    float checker = label_image.at<float>(y,x); //need to look for float and not int as the scalar value is of type double
                    cv::Rect rect;
                    //cout << "check:" << checker << endl;
                    if(checker ==255) {
                        //fill region from a point
                        cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
                        label_count++;
                        //cout << label_image << endl <<"by checking: " << label_image.at<float>(y,x) <<endl;
                        //cout << label_image;

                        //a vector of all points in a blob
                        std::vector<cv::Point> blob;

                        for(int i=rect.y; i < (rect.y+rect.height); i++) {
                            for(int j=rect.x; j < (rect.x+rect.width); j++) {
                                float chk = label_image.at<float>(i,j);
                                //cout << chk << endl;
                                if(chk == label_count-1) {
                                    blob.push_back(cv::Point(j,i));
                                }                       
                            }
                        }
                        //place the points of a single blob in a grouping
                        //a vector of vector points
                        blobs.push_back(blob);
                    }
                }
            }
            cout << label_count <<endl;
            //imshow("label image",label_image);

//vector <vector<Point> >contours_poly(contours.size());

vector<Rect> boundRect(blobs.size());
float skew_val=0.0;
int k=0;
int val[300]={0};
Mat img;
cout<<" ";
for (size_t i = 0; i < blobs.size(); i++) {
// approxPolyDP(Mat(blobs[i]),contours_poly[i],3,true);
 boundRect[i]=boundingRect(Mat( blobs[i]));
  int x1=(boundRect[i].tl()).x;
    int y1=(boundRect[i].tl()).y;
    int x=abs(x1-(boundRect[i].br()).x);
    int sign_x= float(x)/(abs(x1-(boundRect[i].br()).x));
    int y=abs(y1-(boundRect[i].br()).y);
    int sign_y= float(y)/(abs(y1-(boundRect[i].br()).y));
//skew_val=(atan(float(y)/x))*(180/3.14);//(boundRect[i].br()).y - (boundRect[i].tl()).y)/((boundRect[i].br()).x - (boundRect[i].tl()).x));
//cout<<"skew_val="<<skew_val;
rectangle(boss,boundRect[i].tl(),boundRect[i].br(), Scalar(255,0,0),2,8,0);
imshow("rectangle",boss);
if((y>=x)&&(x>0 && y>0))// &&(x*y>=500) &&((boss.cols-x)>=10)&&((boss.rows-y)>=10))
{
//incase of only numbers being detected, use same area for 6 or more
 if(x1+x+4>boss.rows || y1+y+4>boss.cols) {
 	if((sign_x>0)&&(sign_y>0))
            img=label_image(Rect(x1,y1,x,y));
        if((sign_x>0)&&(sign_y<0))
        	img=label_image(Rect(x1,y1-y,x,y));
        if((sign_x<0)&&(sign_y>0))
        	img=label_image(Rect(x1-x,y1,x,y));
        if((sign_x<0)&&(sign_y<0))
        	img=label_image(Rect(x1-x,y1-y,x,y));

       } else {
       	if((sign_x>0)&&(sign_y>0))
            img=label_image(Rect(x1-2,y1-2,x+4,y+4));
        if((sign_x>0)&&(sign_y<0))
        	img=label_image(Rect(x1-2,y1-2-y,x+4,y+4));
        if((sign_x<0)&&(sign_y>0))
        	img=label_image(Rect(x1-2-x,y1-2,x+2,y+4));
        if((sign_x<0)&&(sign_y<0))
        	img=label_image(Rect(x1-x-2,y1-y-2,x+4,y+4));
            //img=label_image(Rect(x1-2, y1-2, x+4,y+4));
       } 
//img=label_image(Rect((boundRect[i].tl()).x-2, (boundRect[i].tl()).y-2, x+4,y+4));
//{
// minEnclosingCircle((matrices)contours_poly[i],center[i],radius[i]);

// tesseract::TessBaseAPI tess;
// tess.Init(NULL, NULL, tesseract::OEM_DEFAULT);
// tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
//cv::threshold(img, img, 0, 255, CV_THRESH_BINARY_INV);
//cout<<"rows="<<img.rows<<"cols="<<img.cols;
////imshow("roi",img);
//Mat image;
//erode( img, img33, element );
//GaussianBlur(img,img,Size(15,15),0,0);
// cv::GaussianBlur(img, image, cv::Size(0, 0), 3);
// cv::addWeighted(img, 1.5, image, -0.5, 0, image);
//char b[50], c[100];
cout<<(50*img.rows)/src.rows<<" ";
val[(50*img.rows)/src.rows]+=1;
//val[i]=(50*img.rows)/src.rows;


}
      }

int max_val=0;
int max_val_idx=0;
for(int i=0;i<50;i++)
{
if(val[i]>max_val)
{
max_val=val[i];
//max_val_idx=i;
}
}
cout<<"max_val="<<max_val<<endl;

for (size_t i = 0; i < blobs.size(); i++) {
// approxPolyDP(Mat(blobs[i]),contours_poly[i],3,true);
 boundRect[i]=boundingRect(Mat( blobs[i]));
int x1=(boundRect[i].tl()).x;
    int y1=(boundRect[i].tl()).y;
    int x=abs(x1-(boundRect[i].br()).x);
    int sign_x= float(x)/(abs(x1-(boundRect[i].br()).x));
    int y=abs(y1-(boundRect[i].br()).y);
    int sign_y= float(y)/(abs(y1-(boundRect[i].br()).y));
//int x=abs((boundRect[i].tl()).x- (boundRect[i].br()).x);
//int y=abs((boundRect[i].tl()).y- (boundRect[i].br()).y);
rectangle(boss,boundRect[i].tl(),boundRect[i].br(), Scalar(255,0,0),2,8,0);
//if((y>=x))// &&(x*y>=500) &&((boss.cols-x)>=10)&&((boss.rows-y)>=10))
//{
	if((y>=x)&&(x>0 && y>0))// &&(x*y>=500) &&((boss.cols-x)>=10)&&((boss.rows-y)>=10))
{
//incase of only numbers being detected, use same area for 6 or more
 if(x1+x+4>boss.rows || y1+y+4>boss.cols) {
 	if((sign_x>0)&&(sign_y>0))
            img=label_image(Rect(x1,y1,x,y));
        if((sign_x>0)&&(sign_y<0))
        	img=label_image(Rect(x1,y1-y,x,y));
        if((sign_x<0)&&(sign_y>0))
        	img=label_image(Rect(x1-x,y1,x,y));
        if((sign_x<0)&&(sign_y<0))
        	img=label_image(Rect(x1-x,y1-y,x,y));

       } else {
       	if((sign_x>0)&&(sign_y>0))
            img=label_image(Rect(x1-2,y1-2,x+4,y+4));
        if((sign_x>0)&&(sign_y<0))
        	img=label_image(Rect(x1-2,y1-2-y,x+4,y+4));
        if((sign_x<0)&&(sign_y>0))
        	img=label_image(Rect(x1-2-x,y1-2,x+2,y+4));
        if((sign_x<0)&&(sign_y<0))
        	img=label_image(Rect(x1-x-2,y1-y-2,x+4,y+4));
            //img=label_image(Rect(x1-2, y1-2, x+4,y+4));
       } 
//incase of only numbers being detected, use same area for 6 or more
//img=label_image(Rect((boundRect[i].tl()).x-2, (boundRect[i].tl()).y-2, x+4,y+4));
//{
// minEnclosingCircle((matrices)contours_poly[i],center[i],radius[i]);

// tesseract::TessBaseAPI tess;
// tess.Init(NULL, NULL, tesseract::OEM_DEFAULT);
// tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
       Mat frame;
cv::GaussianBlur(img, frame, cv::Size(0, 0), 3);
cv::addWeighted(frame, -0.7, img, 1.7, 0, img);
imshow("sharpened",img);
cv::threshold(img, img, 0, 255, CV_THRESH_BINARY_INV);
imshow("final inversion",img);
//cout<<"rows="<<img.rows<<"cols="<<img.cols;
////imshow("roi",img);
Mat image;
//erode( img, img33, element );
//GaussianBlur(img,img,Size(15,15),0,0);
// cv::GaussianBlur(img, image, cv::Size(0, 0), 3);
// cv::addWeighted(img, 1.5, image, -0.5, 0, image);
char b[50], c[100];

if((((float(img.rows))/img.cols) >1.5)&&(((float(img.rows))/img.cols) <5)&&(val[(50*img.rows)/src.rows]-max_val <=2)&&(val[(50*img.rows)/src.rows]-max_val >=-2))///(img.rows*img.cols>=(0.005*src.rows*src.cols)))// for removing the noises
{
//cout<<"area="<<float(img.rows*img.cols)/(src.rows*src.cols)<<endl;
sprintf(b, "roi%d.jpg", k);
imwrite(b,img);
sprintf(c, "convert -units PixelsPerInch roi%d.jpg -density 600 output%d.jpg",k,k);
system(c);
sprintf(b, "tesseract output%d.jpg -psm 10 out%d",k,k);
system(b);
k++;
}
}
}
      //imshow("rajnikanth",boss);

//??ADiTI new code

//for (size_t i = 0; i < blobs.size(); i++) {
// approxPolyDP(Mat(blobs[i]),contours_poly[i],3,true);
// boundRect[i]=boundingRect(Mat( blobs[i]));
//int x=abs((boundRect[i].tl()).x- (boundRect[i].br()).x);
//int y=abs((boundRect[i].tl()).y- (boundRect[i].br()).y);

//if((x>=y) && ((x/y)>=4)&&((x/y)<=9) &&(x*y>=500) &&((boss.cols-x)>=10)&&((boss.rows-y)>=10))
//{
//	k++;
//rectangle(boss,boundRect[i].tl(),boundRect[i].br(), Scalar(255,0,0),2,8,0);
//img=boss(Rect((boundRect[i].tl()).x, (boundRect[i].tl()).y, x,y));

//}
// minEnclosingCircle((matrices)contours_poly[i],center[i],radius[i]);
//}

////imshow("roi///",img);
//imwrite("roi.jpg",img);
//system("tesseract roi.jpg out");
/*if(k==1)
{ cout<<"worked";}
else{
if(k==0)
{cout<<"nothing detected";}
else
{cout<<"too many rectangles";}}
*/


cvWaitKey();
}
 