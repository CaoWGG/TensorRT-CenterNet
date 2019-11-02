//
// Created by cao on 19-10-26.
//

#include <ctdetNet.h>
#include <ctdetLayer.h>
#include <assert.h>
#include <fstream>
static Logger gLogger;

namespace ctdet
{

    ctdetNet::ctdetNet(const std::string &onnxFile, const std::string &calibFile,
            ctdet::RUN_MODE mode):mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),
                                  runMode(mode),runIters(0)
    {

        const int maxBatchSize = 1;
        nvinfer1::IHostMemory *modelStream{nullptr};
        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();


        auto parser = nvonnxparser::createParser(*network, gLogger);
        std::cout << "Begin parsing model..." << std::endl;
        if (!parser->parseFromFile(onnxFile.c_str(), verbosity))
        {
            std::string msg("failed to parse onnx file");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 30);// 1G


        if (runMode== RUN_MODE::INT8)
        {
            //nvinfer1::IInt8Calibrator* calibrator;
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(nullptr);
        }
        else if (runMode == RUN_MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
            builder->setFp16Mode(true);
        }
        // config input shape

        std::cout << "Begin building engine..." << std::endl;
        nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (!engine){
            std::string error_message ="Unable to create engine";
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
        }
        std::cout << "End building engine..." << std::endl;

        // We don't need the network any more, and we can destroy the parser.

        parser->destroy();
        // Serialize the engine, then close everything down.
        modelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        builder->destroy();
        assert(modelStream != nullptr);
        mRunTime = nvinfer1::createInferRuntime(gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);

        assert(mEngine != nullptr);
        modelStream->destroy();
        InitEngine();

    }

    ctdetNet::ctdetNet(const std::string &engineFile)
            :mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),runMode(RUN_MODE::FLOAT32),runIters(0)
    {
        using namespace std;
        fstream file;

        file.open(engineFile,ios::binary | ios::in);
        if(!file.is_open())
        {
            cout << "read engine file" << engineFile <<" failed" << endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        std::cout << "deserializing" << std::endl;
        mRunTime = nvinfer1::createInferRuntime(gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(data.get(), length, nullptr);
        assert(mEngine != nullptr);
        InitEngine();
    }

    void ctdetNet::InitEngine() {
        const int maxBatchSize = 1;
        mContext = mEngine->createExecutionContext();
        assert(mContext != nullptr);
        mContext->setProfiler(&mProfiler);
        int nbBindings = mEngine->getNbBindings();
        assert(nbBindings == 4);
        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings);
        int64_t totalSize = 0;
        for (int i = 0; i < nbBindings; ++i)
        {
            nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
            mBindBufferSizes[i] = totalSize;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);
        }
        outputBufferSize = mBindBufferSizes[1] * 6 ;
        cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
    }

    void ctdetNet::doInference(const void *inputData, void *outputData)
    {
        const int batchSize = 1;
        int inputIndex = 0 ;
        CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[inputIndex], inputData, mBindBufferSizes[inputIndex], cudaMemcpyHostToDevice, mCudaStream));
        mContext->execute(batchSize, &mCudaBuffers[inputIndex]);
        CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
        CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
                         ouputSize,ouputSize,classNum,kernelSize,visThresh);

        CUDA_CHECK(cudaMemcpyAsync(outputData, cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream));

        runIters++ ;
    }
    void ctdetNet::saveEngine(const std::string &fileName)
    {
        if(mEngine)
        {
            nvinfer1::IHostMemory* data = mEngine->serialize();
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                return;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
        }

    }
}