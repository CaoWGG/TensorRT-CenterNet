//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_UTILS_H
#define CTDET_TRT_UTILS_H

#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <numeric>
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "cuda_runtime.h"


#ifndef BLOCK
#define BLOCK 512
#endif
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif



class Profiler : public nvinfer1::IProfiler
{
public:
    struct Record
    {
        float time{0};
        int count{0};
    };
    void printTime(const int& runTimes)
    {
        //std::cout << "========== " << mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }
//        auto old_settings = std::cout.flags();
//        auto old_precision = std::cout.precision();
//        // Output header
//        {
//            std::cout << std::setw(maxLayerNameLength) << layerNameStr << " ";
//            std::cout << std::setw(12) << "Runtime, "
//                << "%"
//                << " ";
//            std::cout << std::setw(12) << "Invocations"
//                << " ";
//            std::cout << std::setw(12) << "Runtime, ms" << std::endl;
//        }
//        for (const auto& elem : mProfile)
//        {
//            std::cout << std::setw(maxLayerNameLength) << elem.first << " ";
//            std::cout << std::setw(12) << std::fixed << std::setprecision(1) << (elem.second.time * 100.0F / totalTime) << "%"
//                << " ";
//            std::cout << std::setw(12) << elem.second.count << " ";
//            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << std::endl;
//        }
//        std::cout.flags(old_settings);
//        std::cout.precision(old_precision);
        std::cout<< " total runtime = " << totalTime/runTimes << " ms " << std::endl;
    }

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
    }
private:
    std::map<std::string, Record> mProfile;
};

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
};

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

struct Box{
    float x1;
    float y1;
    float x2;
    float y2;
};
struct landmarks{
    float x;
    float y;
};
struct Detection{
    //x1 y1 x2 y2
    Box bbox;
    //float objectness;
    int classId;
    float prob;
    landmarks marks[5];
};

extern std::vector<float> prepareImage(cv::Mat& img, const bool& forwardFace);
extern void postProcess(std::vector<Detection> & result,const cv::Mat& img, const bool& forwardFace);
extern void postProcess(std::vector<Detection> & result,const int &img_w ,const int& img_h, const bool& forwardFace);
extern void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color, const bool& forwardFace);
extern cv::Scalar randomColor(cv::RNG& rng);
#endif //CTDET_TRT_UTILS_H
