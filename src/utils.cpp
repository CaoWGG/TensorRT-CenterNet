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

std::vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int channel = ctdet::channel ;
    int inputSize = ctdet::inputSize;

    float scale = min(float(inputSize)/img.cols,float(inputSize)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized,scaleSize,0,0);

    cv::Mat cropped = cv::Mat::zeros(inputSize,inputSize,CV_8UC3);
    Rect rect((inputSize- scaleSize.width)/2, (inputSize-scaleSize.height)/2, scaleSize.width,scaleSize.height);

    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(channel);
    cv::split(img_float, input_channels);

    // normalize
    vector<float> result(inputSize*inputSize*channel);
    auto data = result.data();
    int channelLength = inputSize * inputSize;
    for (int i = 0; i < channel; ++i) {
        Mat normed_channel = (input_channels[i]-ctdet::mean[i])/ctdet::std[i];
        memcpy(data,normed_channel.data,channelLength*sizeof(float));
        data += channelLength;
    }
    return result;
}

void postProcess(std::vector<Detection> & result,const cv::Mat& img)
{
    using namespace cv;
    int inputSize = ctdet::inputSize;
    float scale = min(float(inputSize)/img.cols,float(inputSize)/img.rows);
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
    }
}

cv::Scalar randomColor(cv::RNG& rng) {
    int icolor = (unsigned) rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color )
{
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
    cv::putText(img,label,
                cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/3, 8, 0);
    }
}