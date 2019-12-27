//
// Created by cao on 19-11-7.
//

#ifndef CTDET_TRT_PYTHON_API_H
#define CTDET_TRT_PYTHON_API_H
#include "utils.h"
#include "ctdetConfig.h"
typedef struct
{
    int num;
    Detection *det;
}detResult;
extern "C" void* initNet(char* modelpath);
extern "C" detResult predict(void* net,void* inputData,int img_w,int img_h);
extern "C" void* ndarrayToImage(float * src, long* shape, long* strides);
extern "C" void setDevice(int id);
extern "C" void freeResult(detResult *p);
extern "C" void freeNet(void * p);

#endif //CTDET_TRT_PYTHON_API_H