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
imshow("resized",src);

cvtColor(src,src,CV_BGR2GRAY);
imshow("grayscaled",src);

/*Mat kernel1 = Mat::ones(Size(5, 11), CV_8U);
Mat kernel = Mat::ones(Size(3, 9), CV_8U);

dilate(src,src, kernel);
imshow("dilated",src);

erode(src,src,kernel);
imshow("eroded",src);

Mat im1;
Mat im2,output;
src.copyTo(im1);
org.copyTo(im2);
Mat output2 = im1-im2;
imshow("subtraction",output2);*/

//output2.copyTo(src);


//USE TRESSECA- On all blobs
// Now aspect ratio for numbers
// Mode for areas
// Adaptive thresholding

GaussianBlur(src,src,Size(5,5),0,0);

cv::threshold(src, src, 128, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);	//not using adaptive threshollding etc
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
            imshow("label image",label_image);

//vector <vector<Point> >contours_poly(contours.size());

vector<Rect> boundRect(blobs.size());
Mat img;
for (size_t i = 0; i < blobs.size(); i++) {
// approxPolyDP(Mat(blobs[i]),contours_poly[i],3,true);
 boundRect[i]=boundingRect(Mat( blobs[i]));
rectangle(boss,boundRect[i].tl(),boundRect[i].br(), Scalar(255,0,0),2,8,0);
// minEnclosingCircle((matrices)contours_poly[i],center[i],radius[i]);
}
            imshow("rajnikanth",boss);
/*Mat org1;
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
 */ 
cvWaitKey();
return 0;
}
