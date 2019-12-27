//
// Created by cao on 19-10-26.
//


#include <argparse.h>
#include <string>
#include <iostream>
#include <memory>
#include "ctdetNet.h"
#include "utils.h"


int main(int argc, const char** argv){
    optparse::OptionParser parser;
    parser.add_option("-e", "--input-engine-file").dest("engineFile").set_default("test.engine")
            .help("the path of onnx file");
    parser.add_option("-i", "--input-img-file").dest("imgFile").set_default("test.jpg");
    parser.add_option("-c", "--input-video-file").dest("capFile").set_default("test.h264");
    optparse::Values options = parser.parse_args(argc, argv);
    if(options["engineFile"].size() == 0){
        std::cout << "no file input" << std::endl;
        exit(-1);
    }

    cv::RNG rng(244);
    std::vector<cv::Scalar> color = { cv::Scalar(255, 0,0),cv::Scalar(0, 255,0)};
    //for(int i=0; i<ctdet::classNum;++i)color.push_back(randomColor(rng));


    cv::namedWindow("result",cv::WINDOW_NORMAL);
    cv::resizeWindow("result",1024,768);

    ctdet::ctdetNet net(options["engineFile"]);
    std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);

    cv::Mat img ;
    if(options["imgFile"].size()>0)
    {
        img = cv::imread(options["imgFile"]);
        auto inputData = prepareImage(img,net.forwardFace);

        net.doInference(inputData.data(), outputData.get());
        net.printTime();

        int num_det = static_cast<int>(outputData[0]);
        std::vector<Detection> result;
        result.resize(num_det);
        memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

        postProcess(result,img,net.forwardFace);

        drawImg(result,img,color,net.forwardFace);

        cv::imshow("result",img);
        cv::waitKey(0);
    }

    if(options["capFile"].size()>0){
        cv::VideoCapture cap(options["capFile"]);
        while (cap.read(img))
        {
            auto inputData = prepareImage(img,net.forwardFace);

            net.doInference(inputData.data(), outputData.get());
            net.printTime();

            int num_det = static_cast<int>(outputData[0]);

            std::vector<Detection> result;

            result.resize(num_det);

            memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

            postProcess(result,img,net.forwardFace);

            drawImg(result,img,color,net.forwardFace);

            cv::imshow("result",img);
            if((cv::waitKey(1)& 0xff) == 27){
                cv::destroyAllWindows();
                return 0;
            };

        }

    }

    return 0;
}
