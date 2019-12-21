//
// Created by cao on 19-12-20.
//

#include "DCNv2.hpp"
#include "dcn_v2_im2col_cuda.h"

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasCreate(&handle[n]);
        init[n] = 1;
    }
    return handle[n];
}
inline bool is_CHW(nvinfer1::Dims const& dims) {
    return (dims.nbDims == 3 &&
            dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
            dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
            dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

DCNv2Plugin::DCNv2Plugin(int in_channel,
                         int out_channel,
                         int kernel_H,
                         int kernel_W,
                         int deformable_group,
                         int dilation,
                         int groups,
                         int padding,
                         int stride,
                         nvinfer1::Weights const &weight, nvinfer1::Weights const &bias):_in_channel(in_channel),
                        _out_channel(out_channel),_kernel_H(kernel_H),_kernel_W(kernel_W),_deformable_group(deformable_group),
                         _dilation(dilation),_groups(groups),_padding(padding),_stride(stride),_initialized(false){

    if (weight.type == nvinfer1::DataType::kFLOAT)
    {
        _h_weight.assign((float*)weight.values,(float*)weight.values+weight.count);
    } else { throw std::runtime_error("Unsupported  weight dtype");}

    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        _h_bias.assign((float*)bias.values,(float*)bias.values+bias.count);
    } else { throw std::runtime_error("Unsupported  bias dtype");}

}
int DCNv2Plugin::initialize() {
    if(_initialized) return 0;
    auto _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 3);
    assert(is_CHW(this->getInputDims(0)));
    assert(is_CHW(_output_dims));
    size_t ones_size = _output_dims.d[1]*_output_dims.d[2]* sizeof(float);
    size_t weight_size = _h_weight.size()* sizeof(float);
    size_t bias_size = _h_bias.size()* sizeof(float);
    float *ones_cpu = new float[ones_size/ sizeof(float)];
    for (int i = 0; i < ones_size/ sizeof(float); i++) {
        ones_cpu[i] = 1.0;
    }
    CHECK_CUDA(cudaMalloc((void**)&_d_columns, _in_channel * _kernel_H * _kernel_W * ones_size););
    CHECK_CUDA(cudaMalloc((void**)&_d_ones, ones_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_weight, weight_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_bias, bias_size));
    CHECK_CUDA(cudaMemcpy(_d_ones, ones_cpu, ones_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_weight, _h_weight.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_bias, _h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    delete[] ones_cpu;
    _initialized = true;

    return 0;
}
void DCNv2Plugin::terminate() {
    if (!_initialized) {
        return;
    }
    cudaFree(_d_columns);
    cudaFree(_d_bias);
    cudaFree(_d_weight);
    cudaFree(_d_ones);
    _initialized = false;
}

DCNv2Plugin::~DCNv2Plugin() {
    terminate();
}
bool DCNv2Plugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {

    return (type == nvinfer1::DataType::kFLOAT);
}
nvinfer1::Dims DCNv2Plugin::getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 3);
    nvinfer1::Dims const& input = inputDims[0];
    assert(is_CHW(input));
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    for( int d=0; d<input.nbDims; ++d ) {
        output.type[d] = input.type[d];
        output.d[d] = input.d[d];
    }
    output.d[0] = _out_channel;
    output.d[1] = (output.d[1] + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1 ;
    output.d[2] = (output.d[2] + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1 ;
    return output;
}
size_t DCNv2Plugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int DCNv2Plugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                         cudaStream_t stream) {
    float alpha ,beta;
    int m, n, k;

    cublasHandle_t handle = blas_handle();
    const float* input = static_cast<const float *>(inputs[0]);
    const float* offset = static_cast<const float *>(inputs[1]);
    const float* mask = static_cast<const float *>(inputs[2]);
    float * output = static_cast<float *>(outputs[0]);
    nvinfer1::Dims input_dims = this->getInputDims(0);
    assert(batchSize==1);
    int h = input_dims.d[1];
    int w = input_dims.d[2];
    int height_out = (h + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
    int width_out = (w + 2 * _padding - (_dilation * (_kernel_W - 1) + 1)) / _stride + 1;
    m = _out_channel;
    n = height_out * width_out;
    k = 1;
    alpha = 1.0;
    beta = 0.0;
    /// output  nxm
    /// ones    1xn  T ->> nx1
    /// bias    1xm
    /// ones x bias = nxm
    //  add bias
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_ones, k,
                _d_bias, k,&beta,
                output, n);
    // im2col (offset and mask)
    modulated_deformable_im2col_cuda(stream,input,offset,mask,
                                     1, _in_channel, h, w,
                                     height_out, width_out, _kernel_H, _kernel_W,
                                     _padding, _padding, _stride, _stride, _dilation, _dilation,
                                     _deformable_group, _d_columns);
    m = _out_channel;
    n = height_out * width_out;
    k = _in_channel * _kernel_H * _kernel_W;
    alpha = 1.0;
    beta = 1.0;
    // im2col conv
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_columns, n,
                _d_weight, k,
                &beta,
                output, n);
    return 0;
}
