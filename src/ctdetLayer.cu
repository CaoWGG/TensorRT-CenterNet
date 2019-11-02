#include <ctdetLayer.h>

__device__ float Logist(float data){ return 1./(1. + exp(-data)); }

__global__ void CTdetforward_kernel(const float *hm, const float *reg,const float *wh ,
        float *output,const int w,const int h,const int classes,const int kernerl_size,const float visthresh  ) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w*h) return;
    int padding = kernerl_size/2;
    int offset = - padding /2;
    int stride = w*h;
    int grid_x = idx % w ;
    int grid_y = idx / w ;
    int cls,l,m;
    float c_x,c_y;
    for (cls = 0; cls < classes; ++cls )
    {
        int objIndex = stride * cls + idx;
        float objProb = hm[objIndex];
        float max=-1;
        int max_index =0;
        for(l=0 ;l < kernerl_size ; ++l)
            for(m=0 ; m < kernerl_size ; ++m){
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride*cls;
                int valid = (cur_x>=0 && cur_x < w && cur_y >=0 && cur_y <h );
                float val = (valid !=0 ) ? Logist(hm[cur_index]): -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max ) ?  val: max ;
            }
        objProb = Logist(objProb);
        if((max_index == objIndex) && (objProb > visthresh)){

            int resCount = (int)atomicAdd(output,1);
            //printf("%d",resCount);
            char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);
            c_x = grid_x + reg[idx] ; c_y  = grid_y + reg[idx+stride];
            det->bbox.x1 = (c_x - wh[idx]/2)*4;
            det->bbox.y1 = (c_y - wh[idx+stride]/2)*4 ;
            det->bbox.x2 = (c_x + wh[idx]/2)*4;
            det->bbox.y2 = (c_y + wh[idx+stride]/2)*4;
            det->classId = cls;
            det->prob = objProb;
        }
    }
}


void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h;
    CTdetforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,reg,wh,output,w,h,classes,kernerl_size,visthresh);
}
