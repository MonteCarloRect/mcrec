#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"

__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop);
__device__ int double_totalen(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir);
__device__ int double_mol_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir);

int double_equilibration(gDoublebox gDBox, hDoubleBox doubleBox, gOptions gConf, gMolecula gTop){
    cudaError_t cuErr;
    int* xDim;
    int temp;
    
    xDim = (int*) malloc(deviceCount * sizeof(int));
    for(int curDev = 0; curDev < deviceCount; curDev++){
        //set numbers of block equal to numbers of plate per device
        //numbers of thread equal to maximum numbers of molecules 
        temp = 0;
        for(int i = 0; i < doubleBox.devicePlates[i]; i++){    //get maximum number of molecules
            if(doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]] > temp){
                temp = doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]];
            }
        }
        if(temp == 0){
            xDim[curDev] = 1;
        }
        else{
            if(log2(temp) > 7){
                xDim[curDev] = 512;
            }
            else{
                xDim[curDev] = pow(2,ceil(log2(temp)));
            }
        }
    }
    cudaDeviceSynchronize();    //sync
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //calculate equlibration
        double_equilib_cycle<<<doubleBox.devicePlates[curDev], xDim[curDev]>>>(gDBox, gConf, gTop);
    }
    cudaDeviceSynchronize();    //sync after complit all equlibrations
    return 0;
}

__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop){
    int temp;
    __shared__ int yDim;   //y Dimension of array
    float* tempTotEn;   //temp array for calculate total enegry
    float* tempMolEn;   //temp array for calculate molcule energy
    float* tempTotVir;  //temp array for calculate total virial
    float* tempMolVir;  //temp array for calculate molecule virial
    if(threadIdx.x == 0){
        tempTotEn = (float*) malloc(MAXDIM * sizeof(float)); //enegry of yDim molecules
        tempMolEn = (float*) malloc(MAXDIM * sizeof(float));
    }
    __syncthreads();
    
    //calculate total energy of phases
    if(threadIdx.x == 0){   //shared for whjole block
        yDim = ceilf(gDBox.molNum[blockIdx.x] / blockDim.x);
    }
    
    
    double_totalen(gDBox, gConf, gTop, yDim, tempTotEn, tempMolEn, tempTotVir, tempMolVir);
    if(gDBox.molNum[blockIdx.x] > 0){ //cycle only if plate not empty
        for(int i = 0; i < 1000; i++){  //loop for some (set as a parameter to option file)
            //get random number of molecule for block (plate)
            
            //get random transition
                //move molecule
                
                //change volume
                
                //molecule transition
                
        }
        //check equlibration status
    }
    else{
        //mark block as a compleet
    }
    //free arrays
    free(tempTotEn);
    free(tempMolEn);
}

__device__ int double_totalen(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir){
    int curMol; //current molecule
    //nuber of plate equal block number 
    
    for(int i = 0; i < yDim; i++){  //go 
        //get current molecule number
        curMol = gDBox.fMolOnPlate[blockIdx.x] + threadIdx.x * yDim + i;    //current molecule
        //calculate energy for current molecule
        if(curMol < gDBox.molNum[blockIdx.x]){
            double_mol_energy(gDBox, gConf, gTop, yDim, tempTotEn, tempMolEn, tempTotVir, tempMolVir);
        }
        else{   //epty slot - set energy to zero
            tempTotEn[curMol] = 0.0;
            tempTotVir[curMol] = 0.0;
        }
    }
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    
    
    return 0;
}

__device__ int double_mol_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir){
    
    return 0;
}




