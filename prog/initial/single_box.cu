#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../mcrec.h"
#include <time.h>

//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"

__global__ void single_calc(singleBox* gpuSingleBox, potentialParam* gpuParams,int yDim, mixParam** gpuMixParams){
    int test;
    //calculate initial enegry, pressure and other
    //printf("ololo id %d blockIdx.x  %d blockIdx.y %d blockDim.x \n", threadIdx.x, threadIdx.y, blockDim.x);
    test=single_calc_totenergy(yDim, gpuSingleBox, gpuParams, gpuMixParams);
    //main cycle
    test=gpuSingleBox[0].molNum;
    printf("test %f\n",gpuSingleBox[0].boxLen);
}

__device__ int single_calc_totenergy(int yDim, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams){
    float* en;
    //printf("my_test %d\n",blockIdx.y);
//    printf("my_test id %d blockIdx.x  %d blockIdx.y %d blockDim.x \n", threadIdx.x, threadIdx.y, blockDim.x);
//    en=(float*)malloc(yDim*blockDim.x*sizeof(float));
//    //cudaMalloc(&en, yDim*blockDim.x*sizeof(float));
//    
//    for(int i=0;i<yDim;i++){    //for several molecules per thread
//        int curMol=threadIdx.x+i*blockDim.x;
//        //printf("id %d\n", curMol);
//        for(int j=curMol++;j<yDim*blockDim.x;j++){
//            //calculate energy curMol and j molecules
//            single_calc_one_potential(curMol, j, gpuSingleBox, gpuParams, gpuMixParams);
//        }
//    }
//    //summm potential
    return 0;
}

//calculate potential
__device__ void single_calc_one_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams){
    intra_potential(a,b,gpuSingleBox,gpuParams, gpuMixParams);
    inter_potential();
}

__device__ void intra_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams){
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
    float sig;
    float eps;
    
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
    printf("check \n");
    for(int i=0;i<gpuSingleBox[blockIdx.x].aNum[a];i++){
        for(int j=0;j<gpuSingleBox[blockIdx.x].aNum[b];j++){
            //check LJ potentail
            //calculate r
            xa=gpuSingleBox[blockIdx.x].xa[i]-gpuSingleBox[blockIdx.x].xa[j]+dx;
            ya=gpuSingleBox[blockIdx.x].ya[i]-gpuSingleBox[blockIdx.x].ya[j]+dy;
            dz=gpuSingleBox[blockIdx.x].za[i]-gpuSingleBox[blockIdx.x].za[j]+dz;
            ra=xa*xa+ya*ya+za*za;
            
            eps=gpuMixParams[gpuSingleBox[blockIdx.x].aType[a][i]][gpuSingleBox[blockIdx.x].aType[b][j]].epsilon;
            sig=gpuMixParams[gpuSingleBox[blockIdx.x].aType[a][i]][gpuSingleBox[blockIdx.x].aType[b][j]].sigma;
            ra=sig*sig/ra;
            ra=ra*ra*ra;
            sumE+=4.0*eps*(ra*ra-ra);
            sumV+=4.0*eps*(6.0*ra-12*ra*ra);
        }
    }
    printf(" sumE  %f sum %f\n", sumE, sumV);
}

__device__ void inter_potential(){
    printf("ololo\n");
}



