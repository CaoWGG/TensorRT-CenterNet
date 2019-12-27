//
// Created by cao on 19-10-26.
//
#include "utils.h"
#include "ctdetConfig.h"
#include <sstream>


std::vector<float> prepareImage(cv::Mat& img, const bool& forwardFace)
{


    int channel = ctdet::channel ;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = cv::min(float(input_w)/img.cols,float(input_h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized,scaleSize,0,0);


    cv::Mat cropped = cv::Mat::zeros(input_h,input_w,CV_8UC3);
    cv::Rect rect((input_w- scaleSize.width)/2, (input_h-scaleSize.height)/2, scaleSize.width,scaleSize.height);

    resized.copyTo(cropped(rect));


    cv::Mat img_float;
    if(forwardFace)
        cropped.convertTo(img_float, CV_32FC3, 1.);
    else
        cropped.convertTo(img_float, CV_32FC3,1./255.);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(input_h*input_w*channel);
    auto data = result.data();
    int channelLength = input_h * input_w;
    for (int i = 0; i < channel; ++i) {
        cv::Mat normed_channel = (input_channels[i]-ctdet::mean[i])/ctdet::std[i];
        memcpy(data,normed_channel.data,channelLength*sizeof(float));
        data += channelLength;
    }
    return result;
}

void postProcess(std::vector<Detection> & result,const cv::Mat& img, const bool& forwardFace)
{
    using namespace cv;
    int mark;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = min(float(input_w)/img.cols,float(input_h)/img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
    for(auto&item:result)
    {
        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
        y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        if(forwardFace){
            float x,y;
            for(mark=0;mark<5; ++mark ){
                 x = (item.marks[mark].x - dx) / scale ;
                 y = (item.marks[mark].y - dy) / scale ;
                 x = (x > 0 ) ? x : 0 ;
                 y = (y > 0 ) ? y : 0 ;
                 x = (x < img.cols  ) ? x : img.cols - 1 ;
                 y = (y < img.rows ) ? y  : img.rows - 1 ;
                item.marks[mark].x = x ;
                item.marks[mark].y = y ;
            }
        }
    }
}

void postProcess(std::vector<Detection> & result,const int &img_w ,const int& img_h, const bool& forwardFace)
{


    int mark;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = std::min(float(input_w)/img_w,float(input_h)/img_h);
    float dx = (input_w - scale * img_w) / 2;
    float dy = (input_h - scale * img_h) / 2;
    //printf("%f %f %f %d %d \n",scale,dx,dy,img_w,img_h);
    for(auto&item:result)
    {

        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img_w  ) ? x2 : img_w - 1 ;
        y2 = (y2 < img_h ) ? y2  : img_h - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        if(forwardFace){
            float x,y;
            for(mark=0;mark<5; ++mark ){
                x = (item.marks[mark].x - dx) / scale ;
                y = (item.marks[mark].y - dy) / scale ;
                x = (x > 0 ) ? x : 0 ;
                y = (y > 0 ) ? y : 0 ;
                x = (x < img_w  ) ? x : img_w - 1 ;
                y = (y < img_h ) ? y  : img_h - 1 ;
                item.marks[mark].x = x ;
                item.marks[mark].y = y ;
            }
        }
    }
}

cv::Scalar randomColor(cv::RNG& rng) {
    int icolor = (unsigned) rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color, const bool& forwardFace)
{
    int mark;
    int box_think = (img.rows+img.cols) * .001 ;
    float label_scale = img.rows * 0.0009;
    int base_line ;
    for (const auto &item : result) {
        std::string label;
        std::stringstream stream;
        stream << ctdet::className[item.classId] << " " << item.prob << std::endl;
        std::getline(stream,label);

        auto size = cv::getTextSize(label,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);

        cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                      cv::Point(item.bbox.x2 ,item.bbox.y2),
                      color[item.classId], box_think*2, 8, 0);
        if(!forwardFace){
            cv::putText(img,label,
                    cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                    cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/2, 8, 0);
        }
        if(forwardFace)
        {
            for(mark=0;mark<5; ++mark )
            cv::circle(img, cv::Point(item.marks[mark].x, item.marks[mark].y), 1, cv::Scalar(255, 255, 0), 1);
        }

    }
}