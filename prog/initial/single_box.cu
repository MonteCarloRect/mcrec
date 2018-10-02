#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../mcrec.h"
#include <time.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

__global__ void single_calc(singleBox* gpuSingleBox, potentialParam* gpuParams,int yDim){
    //calculate initial enegry, pressure and other
    //printf("ololo id %d blockIdx.x  %d blockIdx.y %d blockDim.x \n", threadIdx.x, threadIdx.y, blockDim.x);
    single_calc_totenergy(yDim,gpuParams);
    //main cycle
    
}

__device__ void single_calc_totenergy(int yDim, potentialParam* gpuParams){
    //printf("ololo 4");
    float* en;
    en=(float*)malloc(yDim*blockDim.x*sizeof(float));
    //cudaMalloc(&en, yDim*blockDim.x*sizeof(float));
    
    for(int i=0;i<yDim;i++){
        int curMol=threadIdx.x+i*blockDim.x;
        //printf("id %d\n", curMol);
        for(int j=curMol++;j<yDim*blockDim.x;j++){
            //calculate energy  curMol and j molecules
            
        }
    }
}

//calculate potential
__device__ void single_calc_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams){
    
    
}




