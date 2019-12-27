//
// Created by cao on 19-11-7.
//

#include "ctdetNet.h"
#include "python_api.h"

void* initNet(char* modelpath)
{
    ctdet::ctdetNet *net=new ctdet::ctdetNet(modelpath);
    return net;
}

void setDevice(int id)
{
    cudaSetDevice(id);
}


detResult predict(void* net, void * inputData,int img_w,int img_h)
{

    int outputCount = ((ctdet::ctdetNet*)net)->outputBufferSize;
    std::unique_ptr<float[]> outputData(new float[outputCount]);
    ((ctdet::ctdetNet*)net)->doInference(inputData,outputData.get());
    auto output = outputData.get();
    int num_det = static_cast<int>(outputData[0]);
    std::vector<Detection> result;
    result.resize(num_det);
    memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));
    postProcess(result,img_w,img_h,((ctdet::ctdetNet*)net)->forwardFace);
    detResult det;
    det.num=result.size();
    det.det =(Detection*)malloc(result.size()*sizeof(Detection));
    memcpy(det.det,result.data(),result.size()*sizeof(Detection));
    return det;
}

void freeNet(void * p)
{
    delete (ctdet::ctdetNet*)p;
}

void freeResult(detResult *p)
{
    free(p->det);
}

void* ndarrayToImage(float * src, long* shape, long* strides)
{
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];
    int step_c = strides[2];
    int step_h = strides[0]/step_c;
    int step_w = strides[1]/step_c;
    step_c = 1;
    //printf("%d %d %d %d %d %d \n",h,w,c,step_h,step_w,step_c);
    float *data=(float*)malloc(h*w*c* sizeof(float));
    int i, j, k;
    int index1, index2 = 0;
    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){

                index1 = k*w*h + i*w + j;
                index2 = step_h*i + step_w*j + step_c*k;
                data[index1] = src[index2];
                //printf("%d %d %d %d %d %f\n",i,j,k,index1,index2,data[index1]);
            }
        }
    }
    return (void*)data;
}
