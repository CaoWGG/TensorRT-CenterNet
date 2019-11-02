//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETNET_H
#define CTDET_TRT_CTDETNET_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <ctdetConfig.h>
#include <utils.h>

namespace ctdet
{
    enum class RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };

    class ctdetNet
    {
    public:
        ctdetNet(const std::string& onnxFile,
                 const std::string& calibFile,
                 RUN_MODE mode = RUN_MODE::FLOAT32);

        ctdetNet(const std::string& engineFile);

        void saveEngine(const std::string& fileName);

        void doInference(const void* inputData, void* outputData);

        void printTime()
        {
            mProfiler.printTime(runIters) ;
        }

        inline size_t getInputSize() {
            return mBindBufferSizes[0];
        };

        int64_t outputBufferSize;
        bool forwardFace;
    private:

        void InitEngine();

        nvinfer1::IExecutionContext* mContext;
        nvinfer1::ICudaEngine* mEngine;
        nvinfer1::IRuntime* mRunTime;

        RUN_MODE runMode;

        std::vector<void*> mCudaBuffers;
        std::vector<int64_t> mBindBufferSizes;
        void * cudaOutputBuffer;

        cudaStream_t mCudaStream;

        int runIters;
        Profiler mProfiler;
    };

}


#endif //CTDET_TRT_CTDETNET_H
