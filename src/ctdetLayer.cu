#include "ctdetLayer.h"
#include "utils.h"
dim3 cudaGridSize(uint n)
{
    uint k = (n - 1) /BLOCK + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}
__device__ float Logist(float data){ return 1./(1. + exp(-data)); }

__global__ void CTdetforward_kernel(const float *hm, const float *reg,const float *wh ,
        float *output,const int w,const int h,const int classes,const int kernel_size,const float visthresh  ) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w * h * classes) return;
    int padding = (kernel_size - 1) / 2;
    int offset = -padding;
    int stride = w * h;
    int grid_x = idx % w;
    int grid_y = (idx / w) % h;
    int cls = idx/w/h ;
    int  l, m;
    int reg_index = idx - cls*stride;
    float c_x, c_y;
    float objProb = Logist(hm[idx]);
    if (objProb > visthresh) {
        float max = -1;
        int max_index = 0;
        for (l = 0; l < kernel_size; ++l)
            for (m = 0; m < kernel_size; ++m) {
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride * cls;
                int valid = (cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h);
                float val = (valid != 0) ? Logist(hm[cur_index]) : -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max) ? val : max;
            }

        if(idx == max_index){
            int resCount = (int) atomicAdd(output, 1);
            //printf("%d",resCount);
            char *data = (char *) output + sizeof(float) + resCount * sizeof(Detection);
            Detection *det = (Detection *) (data);
            c_x = grid_x + reg[reg_index];
            c_y = grid_y + reg[reg_index + stride];
            det->bbox.x1 = (c_x - wh[reg_index] / 2) * 4;
            det->bbox.y1 = (c_y - wh[reg_index + stride] / 2) * 4;
            det->bbox.x2 = (c_x + wh[reg_index] / 2) * 4;
            det->bbox.y2 = (c_y + wh[reg_index + stride] / 2) * 4;
            det->classId = cls;
            det->prob = objProb;
        }
    }
}


__global__ void CTfaceforward_kernel(const float *hm, const float *wh,const float *reg,const float* landmarks,
                                    float *output,const int w,const int h,const int classes,const int kernel_size,const float visthresh  ) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w*h*classes) return;
    int padding = (kernel_size-1)/2;
    int offset = - padding;
    int stride = w * h;
    int grid_x = idx % w;
    int grid_y = (idx / w) % h;
    int cls = idx/w/h ;
    int reg_index = idx - cls*stride;
    int l,m,mark_id;
    float c_x,c_y,scale_w,scale_h;

    float objProb = hm[idx];
    float max=-1;
    int max_index =0;
    if(objProb > visthresh){
        for(l=0 ;l < kernel_size ; ++l)
            for(m=0 ; m < kernel_size ; ++m){
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride*cls;
                int valid = (cur_x>=0 && cur_x < w && cur_y >=0 && cur_y <h );
                float val = (valid !=0 ) ? hm[cur_index]: -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max ) ?  val: max ;
            }
        if(idx == max_index){
            int resCount = (int)atomicAdd(output,1);
            //printf("%d",resCount);
            char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);
            c_x = (grid_x + reg[reg_index+stride] + 0.5)*4 ; c_y  = (grid_y + reg[reg_index] + 0.5) * 4;
            scale_w =  expf(wh[reg_index+stride]) * 4 ; scale_h  = expf(wh[reg_index]) * 4;
            det->bbox.x1 = c_x - scale_w/2;
            det->bbox.y1 = c_y - scale_h/2 ;
            det->bbox.x2 = c_x + scale_w/2;
            det->bbox.y2 = c_y + scale_h/2;
            det->prob = objProb;
            det->classId = cls;
            for(mark_id=0 ; mark_id < 5 ; ++ mark_id){
                det->marks[mark_id].x = det->bbox.x1 + landmarks[reg_index + (2*mark_id+1)*stride]*scale_w;
                det->marks[mark_id].y = det->bbox.y1 + landmarks[reg_index + (2*mark_id)*stride]*scale_h;
            }
        }
    }
}


void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h * classes;
    CTdetforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,reg,wh,output,w,h,classes,kernerl_size,visthresh);
}

void CTfaceforward_gpu(const float *hm, const float *wh,const float *reg,const float* landmarks,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h * classes;
    CTfaceforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,wh,reg,landmarks,output,w,h,classes,kernerl_size,visthresh);
}
