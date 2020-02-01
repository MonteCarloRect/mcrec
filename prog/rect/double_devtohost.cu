#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"


//transfer data from host to devices
int double_box_host_to_device(options &config, hDoubleBox &doubleBox, gDoublebox &gDBox, gDoublebox &hDBox, gSingleBox &hostData, molecules* initMol, int deviceCount){
    cudaError_t cuErr;
    int sum;
    int sum2;
    int id;
    int id2;
    
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate total number of plates
        hDBox.pltNum = (int*) malloc(deviceCount * sizeof(int));
        for(int i = 0; i < deviceCount; i++){
            hDBox.pltNum[i] = doubleBox.devicePlates[i];
        }
        cuErr = cudaMalloc(&gDBox.pltNum, deviceCount*sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.molNum %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.pltNum, hDBox.pltNum, deviceCount*sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate global number of plates for device
        
        
        //allocate total molecules per plate
        hDBox.molNum = (int*) malloc(doubleBox.devicePlates[curDev] * sizeof(int));
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            hDBox.molNum[i] = doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]];
        }
        cuErr = cudaMalloc(&gDBox.molNum, doubleBox.devicePlates[curDev] * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.molNum %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.molNum, hDBox.molNum, deviceCount*sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //allocate molecules coordinates
        sum = 0;
        sum2 = 0;
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            sum += doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]];  //add molecules in current plate
            //get number of atoms in each molecules
            for(int j = 0; j < doubleBox.molNum[doubleBox.platesPerDevice[curDev][i]]; j++){
                sum2 += initMol[doubleBox.mType[doubleBox.platesPerDevice[curDev][i]][j]].atomNum;
            }
        }
        hDBox.xm = (float*) malloc(sum * sizeof(float));
        hDBox.ym = (float*) malloc(sum * sizeof(float));
        hDBox.zm = (float*) malloc(sum * sizeof(float));
        hDBox.mType = (int*) malloc(sum * sizeof(int));
        
        //allocate numbers of liquid/vapor molecules
        hDBox.nVap = (int*) malloc(doubleBox.devicePlates[curDev] * sizeof(int));
        hDBox.nLiq = (int*) malloc(doubleBox.devicePlates[curDev] * sizeof(int));
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            hDBox.nVap[i] = doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]];
            hDBox.nLiq[i] = doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]];
        }
        cuErr = cudaMalloc(&gDBox.nVap, doubleBox.devicePlates[curDev] * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.nVap %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.nLiq, doubleBox.devicePlates[curDev] * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.nLiq %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.nVap, hDBox.nVap, doubleBox.devicePlates[curDev] * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.nLiq, hDBox.nLiq, doubleBox.devicePlates[curDev] * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        //allocate atoms
        hDBox.xa = (float*) malloc(sum2 * sizeof(float));
        hDBox.ya = (float*) malloc(sum2 * sizeof(float));
        hDBox.za = (float*) malloc(sum2 * sizeof(float));
        
        //allocate GPU molecules data
        cuErr = cudaMalloc(&gDBox.xm, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.xm %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.ym, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.ym %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.zm, sum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.zm %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.mType, sum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.mType %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        //allocate GPU atoms data
        cuErr = cudaMalloc(&gDBox.xa, sum2 * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.xa %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.ya, sum2 * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.ya %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.za, sum2 * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.za %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        //allocate first molecules and first atoms
        hDBox.fMolOnPlate = (int*) malloc(doubleBox.devicePlates[curDev] * sizeof(int));
        hDBox.fAtomOfMol = (int*) malloc(sum * sizeof(int));
        cuErr = cudaMalloc(&gDBox.fMolOnPlate, doubleBox.devicePlates[curDev] * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.fMolOnPlate %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.fAtomOfMol, sum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.fAtomOfMol %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //printf("test21-1 \n");
        //calculate first molecules index in plate\u00d1
        id = 0; //set to zero index for each device for molecule
        id2 = 0;    //set to zero index for each first atopm in molecule
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){
            hDBox.fMolOnPlate[i] = id;
            int curPlate = doubleBox.platesPerDevice[curDev][i];    //current plate
            for(int j = 0; j < doubleBox.molNum[curPlate]; j++){   //over all molecules on plate
                hDBox.xm[id] = doubleBox.xm[curPlate][j];   //[number of plate] [number of molecule]
                hDBox.ym[id] = doubleBox.ym[curPlate][j];
                hDBox.zm[id] = doubleBox.zm[curPlate][j];
                hDBox.mType[id] = doubleBox.mType[curPlate][j];
                doubleBox.gpuIndex[curPlate][j] = id;   //set internal GPU index
                id++;
                hDBox.fAtomOfMol[id] = id2;
                for(int k = 0; k < initMol[doubleBox.mType[curPlate][j]].atomNum; k++){
                    hDBox.xa[id2] = doubleBox.xa[curPlate][j][k];
                    hDBox.ya[id2] = doubleBox.ya[curPlate][j][k];
                    hDBox.za[id2] = doubleBox.za[curPlate][j][k];
                    id2++;
                }
            }
        }
        //printf("test21-2 \n");
        //set liquid/vapor lists
        hDBox.liqList = (int*) malloc(sum * sizeof(int));
        hDBox.vapList = (int*) malloc(sum * sizeof(int));
        cuErr = cudaMalloc(&gDBox.liqList, sum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.liqList %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gDBox.vapList, sum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.vapList %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        id = 0;
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){    //over plates
            int curPlate = doubleBox.platesPerDevice[curDev][i];    //current plate
            for(int j = 0; j < hDBox.nLiq[i]; j++){            //all molecules in liquid
                hDBox.liqList[hDBox.fMolOnPlate[i] + j] = doubleBox.gpuIndex[curPlate][j];
            }
            for(int j = 0; j < hDBox.nVap[i]; j++){            //all molecules in vapor
                hDBox.vapList[hDBox.fMolOnPlate[i] + j] = doubleBox.gpuIndex[curPlate][j];
            }
        }
        
        
        //printf("test21-3 \n");
        //copy molecules data to GPU
        cuErr = cudaMemcpy(gDBox.xm, hDBox.xm, sum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.ym, hDBox.ym, sum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.zm, hDBox.zm, sum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //copy first atoms
        cuErr = cudaMemcpy(gDBox.fMolOnPlate, hDBox.fMolOnPlate, doubleBox.devicePlates[curDev]*sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.fAtomOfMol, hDBox.fMolOnPlate, sum * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //copy atoms to GPU
        cuErr = cudaMemcpy(gDBox.xa, hDBox.xa, sum2 * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.ya, hDBox.xa, sum2 * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.za, hDBox.xa, sum2 * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //equilibrated cycle
        hDBox.eqStep = (int*) malloc(config.plateNum * sizeof(int));
        for(int i = 0; i < config.plateNum; i++){
            hDBox.eqStep[i] = 0;
        }        
        cuErr = cudaMalloc(&gDBox.eqStep, config.plateNum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gDBox.liqList %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gDBox.eqStep, hDBox.eqStep, config.plateNum * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        //free host arrays
        free(hDBox.pltNum);
        free(hDBox.molNum);
        free(hDBox.xm);
        free(hDBox.ym);
        free(hDBox.zm);
        free(hDBox.mType);
        free(hDBox.nVap);
        free(hDBox.nLiq);
        free(hDBox.xa);
        free(hDBox.ya);
        free(hDBox.za);
        free(hDBox.fMolOnPlate);
        free(hDBox.fAtomOfMol);
        free(hDBox.liqList);
        free(hDBox.vapList);
        printf("\n free %d done\n", curDev);
    }
    
    return 1;
}


//first time allocation plates to devices
int double_box_init_allocate(options &config, hDoubleBox &doubleBox, int deviceCount){
    int id;
    //allocate plates evenly
    printf("distribution of plates\n");
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.plateDevice[i] = (int) i * deviceCount / config.plateNum;
        printf(" plate %d device %d\n", i, doubleBox.plateDevice[i]);
    }
    for(int i = 0; i < deviceCount; i++){
        doubleBox.devicePlates[i] = 0;
        for(int j = 0; j < config.plateNum; j++){
            doubleBox.platesPerDevice[i][j] = 0;
        }
    }
    for(int i = 0; i < config.plateNum; i++){
        doubleBox.devicePlates[doubleBox.plateDevice[i]]++;
    }
    for(int i = 0; i < deviceCount; i++){
        id = 0;
        for(int j = 0; j < config.plateNum; j++){
            if(doubleBox.plateDevice[j] == i){
                doubleBox.platesPerDevice[i][id] = j;
                id++;
            }
        }
    }
    
    for(int i = 0; i < deviceCount; i++){
        printf(" device %d calculate %d plates \n ", i, doubleBox.devicePlates[i]);
        for(int j = 0; j < doubleBox.devicePlates[i]; j++){
            printf(" pl  %d   \n", doubleBox.platesPerDevice[i][j]);
        }
    }
    
    return 1;
}


