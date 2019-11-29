//
// Created by cao on 19-10-26.
//
#include <utils.h>
#include <ctdetConfig.h>
#include <sstream>

dim3 cudaGridSize(uint n)
{
    uint k = (n - 1) /BLOCK + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}

std::vector<float> prepareImage(cv::Mat& img, const bool& forwardFace)
{

    int channel = ctdet::channel ;
    int inputSize = ctdet::inputSize;

    float scale = std::min(float(inputSize)/img.cols,float(inputSize)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized,scaleSize,0,0);

    cv::Mat cropped = cv::Mat::zeros(inputSize,inputSize,CV_8UC3);
    cv::Rect rect((inputSize- scaleSize.width)/2, (inputSize-scaleSize.height)/2, scaleSize.width,scaleSize.height);

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
    std::vector<float> result(inputSize*inputSize*channel);
    auto data = result.data();
    int channelLength = inputSize * inputSize;
    for (int i = 0; i < channel; ++i) {
        cv::Mat normed_channel = (input_channels[i]-ctdet::mean[i])/ctdet::std[i];
        memcpy(data,normed_channel.data,channelLength*sizeof(float));
        data += channelLength;
    }
    return result;
}

void postProcess(std::vector<Detection> & result,const cv::Mat& img, const bool& forwardFace)
{

    int mark;
    int inputSize = ctdet::inputSize;
    float scale = std::min(float(inputSize)/img.cols,float(inputSize)/img.rows);
    float dx = (inputSize - scale * img.cols) / 2;
    float dy = (inputSize - scale * img.rows) / 2;
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
    int inputSize = ctdet::inputSize;
    float scale = std::min(float(inputSize)/img_w,float(inputSize)/img_h);
    float dx = (inputSize - scale * img_w) / 2;
    float dy = (inputSize - scale * img_h) / 2;
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
                      color[item.classId], box_think, 8, 0);
        if(!forwardFace){
            cv::putText(img,label,
                    cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                    cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/3, 8, 0);
        }
        if(forwardFace)
        {
            for(mark=0;mark<5; ++mark )
            cv::circle(img, cv::Point(item.marks[mark].x, item.marks[mark].y), 2, cv::Scalar(255, 255, 0), 2);
        }

    }
}