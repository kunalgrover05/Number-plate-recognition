#include <iostream>
#include "opencv/cv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <ctime>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <string> 
 using namespace std;
 using namespace cv;

int main( int argc, char** argv )
{
clock_t t1,t2;
t1=clock();

cvUseOptimized(true);

Mat src; 
Mat gray;
Mat binary, label_image;

char p[100];

sprintf(p,"convert %s -units PixelsPerInch -density 300 input.jpg", argv[1]);
system(p);
src=imread("input.jpg");

int times;

Mat org;
src.copyTo(org);
Mat hsv;

//improving contrast
for(int x_coor=0;x_coor<src.cols;x_coor++) {
     for(int y_coor=0;y_coor<src.rows;y_coor++) {
         if(src.at<Vec3b>(y_coor,x_coor)[0]>180 && src.at<Vec3b>(y_coor,x_coor)[1]>180 && src.at<Vec3b>(y_coor,x_coor)[2]>180)
         	              src.at<Vec3b>(y_coor,x_coor)[1]=255,src.at<Vec3b>(y_coor,x_coor)[2]=255, src.at<Vec3b>(y_coor,x_coor)[0]=255;

         if(src.at<Vec3b>(y_coor,x_coor)[0]<50 && src.at<Vec3b>(y_coor,x_coor)[1] <50 && src.at<Vec3b>(y_coor,x_coor)[2] <50)
              src.at<Vec3b>(y_coor,x_coor)[1]=0,src.at<Vec3b>(y_coor,x_coor)[2]=0, src.at<Vec3b>(y_coor,x_coor)[0]=0;
     }
 }
imshow("increased contrast",src);

cvtColor(src,hsv,CV_BGR2HSV);

Mat conv;
cvtColor(hsv,conv,CV_HSV2BGR);
imshow("illumination correction",conv);
//char s[100];
cvtColor(conv,gray,CV_BGR2GRAY);
cv::threshold(gray, binary, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);   
imshow("thresholding",binary);
Mat binaryD,binaryM,binaryD4;
Mat kernel = Mat::ones(Size(3,3), CV_8U);
Mat kernel_s = Mat::ones(Size(1,1), CV_8U);

dilate(binary,binary,kernel);
erode(binary, binaryD,kernel_s);
medianBlur(binaryD,binaryM,1);
imshow("binaryM",binaryM);

Mat binary2=cv::Scalar::all(255)-binaryM;
dilate(binary2, binary2,kernel_s);
imshow("binary2",binary2);

Mat kernel4=Mat::ones(Size(2,6),CV_8U); 

binaryM.copyTo(binaryD4);
imshow("binaryD4",binaryD4);

vector < vector<cv::Point>  > blobs;
            blobs.clear();

            // Fill the label_image with the blobs
            // 0  - background
            // 1  - unlabelled foreground
            // 2+ - labelled foreground

            ///input is a binary image therefore values are either 0 or 1
            ///out objective is to find a set of 1's that are together and assign 2 to it
            ///then look for other 1's, and assign 3 to it....so on a soforth

            binaryD4.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S! Because the CV::SCALAR is a double value in the function floodfill

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

Mat img;
imshow("label_image",label_image);
int val[20]={0};
int val_y[20]={0};
int H=src.rows;
int W=src.cols;

vector<Rect> boundRect(blobs.size());
int k = 0;

Point p1,p2;
int x1,y1,x,y;

for (size_t i = 0; i < blobs.size(); i++) {
    boundRect[i]=boundingRect(Mat( blobs[i]));
    p1=boundRect[i].tl();
    p2=boundRect[i].br();
    x1=p1.x;
    y1=p1.y;
    x=-x1+p2.x;
    y=-y1+p2.y;
    if(y>H/32 && (y>=x) && ((x>0 && y>0)))// && y1>=org.rows/3 && p2.y>=org.rows/3) 
{   
        rectangle(org,p1,p2,Scalar(0,0,255),2,8,0);
        val[(20*y)/H]++;
        val_y[(y1*20)/H]++;
    }
}


int max_val=0;
int max_val_idx=0;
int max_y_idx=0;
int max_y=0;
for(int i=0;i<20;i++) {
    // cout<<" "<<val_y[i];
    if(val_y[i]>max_y)
    {
        max_y=val_y[i];
        max_y_idx=i;
    }
    if(val[i]>max_val) {
        max_val=val[i];
        max_val_idx=i;
    }
}

dilate(binary2,binary2,kernel_s);

cv::threshold(label_image,label_image, 0, 255, CV_THRESH_BINARY_INV);
char b[100];

for (size_t i = 0; i < blobs.size(); i++) {
    boundRect[i]=boundingRect(Mat( blobs[i]));
    Point p1=boundRect[i].tl();
    Point p2=boundRect[i].br();

    int x1=p1.x;
    int y1=p1.y;
    int x=abs(x1-p2.x);
    int y=abs(y1-p2.y);
    if(y>H/32 && (y>=x)&&(x>0 && y>0))// && y1>=org.rows/3 && p2.y>=org.rows/3)//&&(x*y>=100)&&(x*y<=700))// &&(x*y>=500) &&((boss.cols-x)>=10)&&((boss.rows-y)>=10))
    {

        if( y/x<=4 && abs((20*y)/H-max_val_idx)<=1 && abs((y1*20)/H-max_y_idx)<=1)// && x*y>=H/2 && x*y<=15*H/2 )
        {  

            if(x1+x+4>W || y1+y+4>H  || x1<2 || y1<2) {
               img=binary2(Rect(x1,y1,x,y));
           } else {
                img=binary2(Rect(x1-1, y1-1, x+2,y+2));
            }
            rectangle(org,p1,p2,Scalar(0,255,0),2,8,0);
           // cout<<"area="<<x*y<<endl;
            sprintf(b, "roi%d.tif", k);
            // imwrite(b,img);
            // sprintf(c, "convert -units PixelsPerInch roi%d.tiff -density 600 output%d.tiff", k, k);
            // popen(c,"r");
            sprintf(b, "tesseract roi%d.tif -psm 10 out%d num_plate",k,k);
            popen(b,"r");
            k++;
        }
    }
}
 
 imshow("Final Image", org);
 sprintf(b, "output%s", argv[1]);
 cout<<b;
 imwrite(b,org);

t2=clock();
cout<<"time"<<((float)(t2-t1))/CLOCKS_PER_SEC;
cvWaitKey();
return 0;


}

    
