//
// Created by cao on 19-10-26.
//


#include <argparse.h>
#include <string>
#include <iostream>
#include "ctdetNet.h"
#include "utils.h"

int main(int argc, const char** argv)
{
    optparse::OptionParser parser;
    parser.add_option("-i", "--input-onnx-file").dest("onnxFile")
            .help("the path of onnx file");
    parser.add_option("-o", "--output-engine-file").dest("outputFile")
            .help("the path of engine file");
    parser.add_option("-m", "--mode").dest("mode").set_default<int>(0)
            .help("run-mode, type int");
    parser.add_option("-c", "--calib").dest("calibFile").help("calibFile, type str");
    optparse::Values options = parser.parse_args(argc, argv);
    if(options["onnxFile"].size() == 0){
        std::cout << "no file input" << std::endl;
        exit(-1);
    }
    ctdet::RUN_MODE mode = ctdet::RUN_MODE::FLOAT32;
    if(options["mode"] == "0" ) mode = ctdet::RUN_MODE::FLOAT32;
    if(options["mode"] == "1" ) mode = ctdet::RUN_MODE::FLOAT16;
    if(options["mode"] == "2" ) mode = ctdet::RUN_MODE::INT8;

    ctdet::ctdetNet net(options["onnxFile"], options["calibFile"] ,mode);
    net.saveEngine(options["outputFile"]);

    std::cout << "save  " << options["outputFile"] <<std::endl;
}