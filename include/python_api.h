//
// Created by cao on 19-11-7.
//

#ifndef CTDET_TRT_PYTHON_API_H
#define CTDET_TRT_PYTHON_API_H
#include <utils.h>
typedef struct
{
    int num;
    Detection *det;
}detResult;
extern "C" void* init_Net(char* modelpath);
extern "C" detResult predict(void* net,void* inputData,int img_w,int img_h);
extern "C" void* ndarray_to_image(unsigned char * src, long* shape, long* strides);
extern "C" void setdevice(int id);
extern "C" void free_result(detResult *p);

#endif //CTDET_TRT_PYTHON_API_H
