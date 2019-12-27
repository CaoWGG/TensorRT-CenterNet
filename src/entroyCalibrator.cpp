//
// Created by cao on 19-12-16.
//

#include "entroyCalibrator.h"
#include "utils.h"
#include "ctdetConfig.h"
#include <fstream>
#include <iterator>
namespace nvinfer1
{

    int8EntroyCalibrator::int8EntroyCalibrator(const int &bacthSize, const std::string &imgPath,
        const std::string &calibTablePath):batchSize(bacthSize),calibTablePath(calibTablePath),imageIndex(0),forwardFace(
            false){
        int inputChannel = ctdet::channel;
        int inputH = ctdet::input_h;
        int inputW = ctdet::input_w;
        inputCount = bacthSize*inputChannel*inputH*inputW;
        std::fstream f(imgPath);
        if(f.is_open()){
            std::string temp;
            while (std::getline(f,temp)) imgPaths.push_back(temp);

        }
        batchData = new float[inputCount];
        if(ctdet::className[0]=="face")
        {
            forwardFace = true;
            std::cout << "use centerface model," <<std::endl ;
        }
        if (forwardFace) std::cout << "forwardFace : true" << std::endl;
        else std::cout << "forwardFace : false" << std::endl;
        CUDA_CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
    }

    int8EntroyCalibrator::~int8EntroyCalibrator() {
        CUDA_CHECK(cudaFree(deviceInput));
        if(batchData)
            delete[] batchData;
    }

    bool int8EntroyCalibrator::getBatch(void **bindings, const char **names, int nbBindings){
        if (imageIndex + batchSize > int(imgPaths.size()))
            return false;
        // load batch
        float* ptr = batchData;
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
        {
            auto img = cv::imread(imgPaths[j]);
            auto inputData = prepareImage(img,forwardFace);
            if (inputData.size() != inputCount)
            {
                std::cout << "InputSize error. check include/ctdetConfig.h" << std::endl;
                return false;
            }
            //assert(inputData.size() == inputCount);
            memcpy(ptr,inputData.data(),inputData.size()*sizeof(float));

            ptr += inputData.size();
            std::cout << "load image " << imgPaths[j] << "  " << (j+1)*100./imgPaths.size() << "%" << std::endl;
        }
        imageIndex += batchSize;
        CUDA_CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = deviceInput;
        return true;
    }
    const void* int8EntroyCalibrator::readCalibrationCache(std::size_t &length)
    {
        calibrationCache.clear();
        std::ifstream input(calibTablePath, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                    std::back_inserter(calibrationCache));

        length = calibrationCache.size();
        return length ? &calibrationCache[0] : nullptr;
    }

    void int8EntroyCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
    {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
}