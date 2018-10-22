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
    single_calc_totenergy(yDim,gpuSingleBox,gpuParams);
    //main cycle
    
}

__device__ void single_calc_totenergy(int yDim, singleBox* gpuSingleBox, potentialParam* gpuParams){
    //printf("ololo 4");
    float* en;
    en=(float*)malloc(yDim*blockDim.x*sizeof(float));
    //cudaMalloc(&en, yDim*blockDim.x*sizeof(float));
    
    for(int i=0;i<yDim;i++){    //for several molecules per thread
        int curMol=threadIdx.x+i*blockDim.x;
        //printf("id %d\n", curMol);
        for(int j=curMol++;j<yDim*blockDim.x;j++){
            //calculate energy curMol and j molecules
            single_calc_one_potential(curMol,j,gpuSingleBox,gpuParams);
        }
    }
}

//calculate potential
__device__ void single_calc_one_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams){
    intra_potential(a,b,gpuSingleBox,gpuParams);
    inter_potential();
}

__device__ void intra_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuMixParams){
    //Lennard-Jones potential
    float sumE; //energy
    float sumV;    //virial 
    float ra;   //distance beetwin atoms 
    float xa;   //
    float ya;
    float za;
    float dx;   //
    float dy;
    float dz;
    
    sumE=0;
    sumV=0;
    //check length/2
    //X
    if(gpuSingleBox[blockIdx.x].xm[a]-gpuSingleBox[blockIdx.x].xm[b]>0.5*gpuSingleBox[blockIdx.x].boxLen){
        dx=gpuSingleBox[blockIdx.x].xm[a]-gpuSingleBox[blockIdx.x].xm[b]-gpuSingleBox[blockIdx.x].boxLen;
    }
    else if(gpuSingleBox[blockIdx.x].xm[a]-gpuSingleBox[blockIdx.x].xm[b]<-0.5*gpuSingleBox[blockIdx.x].boxLen){
        dx=(gpuSingleBox[blockIdx.x].xm[a]-gpuSingleBox[blockIdx.x].xm[b])+gpuSingleBox[blockIdx.x].boxLen;
    }
    else{
        dx=gpuSingleBox[blockIdx.x].xm[a]-gpuSingleBox[blockIdx.x].xm[b];
    }
    //Y
    if(gpuSingleBox[blockIdx.x].ym[a]-gpuSingleBox[blockIdx.x].ym[b]>0.5*gpuSingleBox[blockIdx.x].boxLen){
        dy=gpuSingleBox[blockIdx.x].ym[a]-gpuSingleBox[blockIdx.x].ym[b]-gpuSingleBox[blockIdx.x].boxLen;
    }
    else if(gpuSingleBox[blockIdx.x].ym[a]-gpuSingleBox[blockIdx.x].ym[b]<-0.5*gpuSingleBox[blockIdx.x].boxLen){
        dy=gpuSingleBox[blockIdx.x].ym[a]-gpuSingleBox[blockIdx.x].ym[b]+gpuSingleBox[blockIdx.x].boxLen;
    }
    else{
        dy=gpuSingleBox[blockIdx.x].ym[a]-gpuSingleBox[blockIdx.x].ym[b];
    }
    //Z
    if(gpuSingleBox[blockIdx.x].zm[a]-gpuSingleBox[blockIdx.x].zm[b]>0.5*gpuSingleBox[blockIdx.x].boxLen){
        dz=gpuSingleBox[blockIdx.x].zm[a]-gpuSingleBox[blockIdx.x].zm[b]-gpuSingleBox[blockIdx.x].boxLen;
    }
    else if(gpuSingleBox[blockIdx.x].zm[a]-gpuSingleBox[blockIdx.x].zm[b]<-0.5*gpuSingleBox[blockIdx.x].boxLen){
        dz=gpuSingleBox[blockIdx.x].zm[a]-gpuSingleBox[blockIdx.x].zm[b]+gpuSingleBox[blockIdx.x].boxLen;
    }
    else{
        dz=gpuSingleBox[blockIdx.x].zm[a]-gpuSingleBox[blockIdx.x].zm[b];
    }
    
    for(int i=0;i<gpuSingleBox[blockIdx.x].aNum[a];i++){
        for(int j=0;j<gpuSingleBox[blockIdx.x].aNum[b];j++){
            //check LJ potentail
            //calculate r
            xa=gpuSingleBox[blockIdx.x].xa[i]-gpuSingleBox[blockIdx.x].xa[j]-1;
            sumE+=1;
        }
    }
    
}

__device__ void inter_potential(){
    
}



