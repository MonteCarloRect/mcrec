#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"

__device__ int single_conf_change(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox, curandState &devStates);
__device__ int single_one_energy(int yDim, int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox);

//
//#include <curand_kernel.h>
//#include <cuda.h>

//

//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"

__global__ void single_calc(int yDim, gOptions gConf, gMolecula gTop, gSingleBox gBox){
    int test;
    __shared__ int randMol;
    curandState devStates;
    //calculate initial enegry, pressure and other
    //printf("ololo id %d blockIdx.x  %d blockIdx.y %d blockDim.x \n", threadIdx.x, threadIdx.y, blockDim.x);
    //test=single_calc_totenergy(yDim, gpuSingleBox, gpuParams, gpuMixParams);
    //main cycle
    //test=gpuSingleBox[0].molNum;
    test=1;
    //printf("deb 81 %d \n", gTop.aNum[0]);
    single_calc_totenergy(yDim, gConf, gTop, gBox);
    __syncthreads();
    curand_init(1234, threadIdx.x, 0, &devStates);
    //loop for change molecules position
    for(int i=0; i< 10; i++){
        //get random numbers
        if(threadIdx.x==0){
            //gBox.curMol[blockIdx.x] = (curand_uniform(&devStates)*gBox.molNum[blockIdx.x]);
            randMol = curand_uniform(&devStates)*gBox.molNum[blockIdx.x];
            printf("random %d %d  \n", i, randMol);
            //change molecul
        }
        __syncthreads();
        //------calculate old energy
        
        if(threadIdx.x==0){
            single_conf_change(randMol, gConf, gTop, gBox, devStates);
        }
        __syncthreads();
        
        //-------calculate new energy
        //calculate properties
    }
        


    //printf("test test %f  %d\n", gBox.xm[threadIdx.x], threadIdx.x);
}

__device__ int single_calc_totenergy(int yDim, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    float* en;
    int curMol;
    int curMol2;
    __shared__ int maxMol;
    float sumE;
    float sumV;
    __shared__ int reduce;
    //printf("tot energy %d\n",blockIdx.x);
    //printf("my_test id %d blockIdx.x  %d blockIdx.y %d blockDim.x \n", threadIdx.x, threadIdx.y, blockDim.x);
//    en=(float*)malloc(yDim*blockDim.x*sizeof(float));
//    //cudaMalloc(&en, yDim*blockDim.x*sizeof(float));
//    
    maxMol = gBox.fMol[blockIdx.x]+blockDim.x*yDim;
    for(int i=0; i<yDim; i++){    //for several molecules per thread
        curMol=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x;    //current number of molecule
        //printf("yDim %d i %d blockDim.x %d curmol %d maxmol %d fmol %d \n", yDim, i, blockDim.x, curMol, maxMol, gBox.fMol[blockIdx.x]);
        gBox.mEnergy[curMol]=0.0;
        gBox.mVirial[curMol]=0.0;
        for(int j= curMol++; j < maxMol; j++){    //curMol+1 //gBox.fMol[blockIdx.x]+blockDim.x*yDim
            //calculate energy curMol and j molecules
            //printf("test a %d b %d\n", curMol, j);
            single_calc_one_potential(curMol, j, gConf, gTop, gBox, sumE, sumV);
            //sum+=1;
        }
        //printf(" thread %d cur mol %d sum %f\n", threadIdx.x, curMol, sum);
        //gBox.mEnergy[curMol]=1.0;
        //printf("curmol %d energy %f virial %f \n", curMol, gBox.mEnergy[curMol], gBox.mVirial[curMol]);
        //printf("curmol %d energy %f virial %f \n", curMol, sumE, sumV);
    }
    __syncthreads();
//  //summm potential   //change reduse
    if(threadIdx.x == 0){
        gBox.energy[blockIdx.x]=0.0;
        gBox.virial[blockIdx.x]=0.0;
        for(int i=gBox.fMol[blockIdx.x]; i<maxMol; i++){
            gBox.energy[blockIdx.x]+=gBox.mEnergy[i];
            gBox.virial[blockIdx.x]+=gBox.mVirial[i];
        }
        printf("variant 1 total enegry %f virial %f \n", gBox.energy[blockIdx.x], gBox.virial[blockIdx.x]);
    }
    __syncthreads();
// variant 2
    reduce = blockDim.x / 2;
    gBox.energy[blockIdx.x]=0.0;
    gBox.virial[blockIdx.x]=0.0;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            curMol2=gBox.fMol[blockIdx.x]+threadIdx.x;
            for(int i=0; i< yDim; i++){
                curMol=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x + reduce;
                gBox.mEnergy[curMol2]+=gBox.mEnergy[curMol];
                gBox.mVirial[curMol2]+=gBox.mVirial[curMol];
            }
        //printf(" reduce %d \n", reduce);
        }
        __syncthreads();
        reduce = reduce / 2;
        //
    }
    if(threadIdx.x == 0){
        gBox.energy[blockIdx.x] = gBox.mEnergy[0];
        gBox.virial[blockIdx.x] = gBox.mVirial[0];
        printf("variant 2 total enegry %f virial %f \n", gBox.energy[blockIdx.x], gBox.virial[blockIdx.x]);
    }

    return 0;
}

//calculate potential
__device__ int single_calc_one_potential(int a, int b, gOptions gConf, gMolecula gTop, gSingleBox &gBox, float &En, float &Vir){
    //printf("one potential a %d b %d \n", a, b);
    intra_potential(a, gConf, gTop, gBox);
    inter_potential(a, b, gConf, gTop, gBox, En, Vir);
    
    return 0;
}

__device__ int inter_potential(int a, int b, gOptions gConf, gMolecula gTop, gSingleBox &gBox, float &En, float &Vir){
    //Lennard-Jones potential
    float sumE; //energy
    float sumV;    //virial 
    float ra;   //distance beetwin atoms 
    float xa;   //
    float ya;
    float za;
    
    float dx;   //distance between molecules 
    float dy;
    float dz;
    float rcut;
    float rtest;
    
    //molecule number
    int curAtomNumA;
    int curAtomNumB;
    int id;
    
    //printf(" a %d type %d b %d typeb %d\n", a ,gBox.mType[a], b, gBox.mType[b]);
    
    curAtomNumA=gTop.aNum[gBox.mType[a]];
    curAtomNumB=gTop.aNum[gBox.mType[b]];
    sumE=0;
    sumV=0;
    rcut=0.5*gBox.boxLen[blockIdx.x];
    //printf("rcut %f\n", rcut);
    
    //molecule periodic boundary condition
    dx=(gBox.xm[b] - gBox.xm[a]);
    dy=(gBox.ym[b] - gBox.ym[a]);
    dz=(gBox.zm[b] - gBox.zm[a]);
    if(dx > rcut){  //
        dx = -rcut*2.0f + dx;
    }
    if(dy > rcut ){
        dy = -rcut*2.0f + dy;
    }
    if(dz > rcut){
        dz = -rcut*2.0f + dz;
    }
    if(dx < -rcut){ //
        dx = rcut*2.0f + dx;
    }
    if(dy < -rcut){
        dy = rcut*2.0f + dy;
    }
    if(dz < -rcut){
        dz = rcut*2.0f + dz;
    }
    //printf("test a %d b %d Na %d Nb %d  faa %d fab %d\n", a, b, curAtomNumA, curAtomNumB, gBox.fAtom[a], gBox.fAtom[b]);
    
    for(int i=0; i<curAtomNumA; i++){
        for(int j=0; j<curAtomNumB; j++){
            id = gBox.aType[gBox.fAtom[b] + j] * gConf.potNum[0] + gBox.aType[gBox.fAtom[a] + i];
            //printf("type a %d b %d testvar \n", gBox.aType[gBox.fAtom[b] + j], gBox.fAtom[a]);
            //id=0;
            
            xa = gBox.xa[gBox.fAtom[b] + j] - gBox.xa[gBox.fAtom[a] - i] + dx;
            ya = gBox.ya[gBox.fAtom[b] + j] - gBox.ya[gBox.fAtom[a] - i] + dy;
            za = gBox.za[gBox.fAtom[b] + j] - gBox.za[gBox.fAtom[a] - i] + dz;
            ra = xa * xa + ya * ya + za * za;
            if(ra > rcut*rcut){
                //rtest= sqrt(ra);
                ra = gTop.sigma[id] * gTop.sigma[id] / ra;
                ra = ra * ra * ra;
                //calculate potential
                
                sumE += gTop.epsi[id] * (ra* ra - ra);
                sumV += gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra);
            }
        }
    }
    gBox.mEnergy[a]+=sumE;
    //__syncthreads();
    gBox.mVirial[a]+=sumV;
    //printf("a %d b %d ra %f rm %f E %f V %f\n", a, b, rtest, sqrt(dx*dx+dy*dy+dz*dz), sumE, sumV);
//    En+=sumE;
//    Vir+=sumV;
    return 0;
}

__device__ int intra_potential(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    return 0;
}

__device__ int single_conf_change(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox, curandState &devStates){
    float dx, dy, dz;
    //get type of change
    
    //molecule move
    dx = curand_uniform(&devStates);
    dy = curand_uniform(&devStates);
    dz = curand_uniform(&devStates);
    gBox.xm[a]+=dx*gBox.transMaxMove[blockIdx.x];
    gBox.ym[a]+=dy*gBox.transMaxMove[blockIdx.x];
    gBox.zm[a]+=dz*gBox.transMaxMove[blockIdx.x];
    return 0;
}

__device__ int single_one_energy(int yDim, int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    __shared__ int maxMol;
    int curMol;
    float sumE;
    float sumV;
    __shared__ int reduce;
    
    maxMol = gBox.fMol[blockIdx.x]+blockDim.x*yDim;
    for(int i=0; i<yDim; i++){    //for several molecules per thread
        curMol = gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x;    //current number of molecule
        gBox.mEnergy[curMol]=0.0;
        gBox.mVirial[curMol]=0.0;
        if(a!=curMol){
            single_calc_one_potential(a, curMol, gConf, gTop, gBox, sumE, sumV);
        }
    }
}
