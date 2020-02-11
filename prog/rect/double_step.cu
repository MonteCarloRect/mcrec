#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"

__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop);
__device__  int double_totalen(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir);
__device__ int double_mol_pair_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempTotVir, int fmol, int smol, float rcut);
__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir, int mol);

///////

int double_equilibration(gDoublebox* &gDBox, hDoubleBox doubleBox, gOptions* gConf, gMolecula* gTop){
    cudaError_t cuErr;
    int* xDim;
    int temp;
    
    xDim = (int*) malloc(deviceCount * sizeof(int));
    for(int curDev = 0; curDev < deviceCount; curDev++){
        //set numbers of block equal to numbers of plate per device
        //numbers of thread equal to maximum numbers of molecules 
        temp = 0;
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){    //get maximum number of molecules in liquib phase
            if(doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]] > temp){
                temp = doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]];
            }
        }
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){    //get maximum numbers of molecules in vapor phase
            if(doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]] > temp){
                temp = doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]];
            }
        }
        if(temp == 0){
            xDim[curDev] = 1;
        }
        else{
            if(log2(temp) > 7){
                xDim[curDev] = MAXDIM; //set blocksize as maximum dimension
            }
            else{
                xDim[curDev] = pow(2,ceil(log2(temp))); //set block size 
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
        double_equilib_cycle<<<doubleBox.devicePlates[curDev], xDim[curDev]>>>(gDBox[curDev], gConf[curDev], gTop[curDev]);
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
        tempTotEn = (float*) malloc(MAXDIM * gridDim.x * sizeof(float)); //enegry of yDim molecules
        tempMolEn = (float*) malloc(MAXDIM * gridDim.x * sizeof(float));
    }
    
    //calculate total energy of phases
    if(threadIdx.x == 0){   //shared for whjole block
        yDim = ceilf(gDBox.molNum[blockIdx.x] / blockDim.x);
    }
    __syncthreads();
    
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
    free(tempTotVir);
    free(tempMolVir);
}

__device__ int double_totalen(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir){
    int curMol; //current molecule
    int curMol2;    //second molecule
    int curId;  //current index
    int reduce;
    //nuber of plate equal block number 
    
    //======calculate liquid phase energy/virial
    tempTotEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;  //set energy to zero
    tempTotVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0; //set virial to zero
    for(int i = 0; i < yDim; i++){
        curId = threadIdx.x * yDim + i; //current id of molecule in liqud list
        if(curId < gDBox.nLiq[blockIdx.x]){
            curMol = gDBox.fMolOnPlate[blockIdx.x] + gDBox.liqList[curId];  //get curent molecule GPU index
            //set here intermolecular potential
            
            //calculate inermolecullar interaction
            for(int j = i+1; j < gDBox.nLiq[blockIdx.x]; j++){
                curMol2 = gDBox.fMolOnPlate[blockIdx.x] + gDBox.liqList[j];
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, tempTotEn, tempTotVir, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
            }
        }
        else{
            //set energy to zero (plase not used)
            tempTotEn[blockIdx.x * MAXDIM + threadIdx.x] += 0.0;
            tempTotVir[blockIdx.x * MAXDIM + threadIdx.x] += 0.0;
        }
    }
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    reduce = blockDim.x / 2;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            tempTotEn[blockIdx.x * MAXDIM + threadIdx.x] += tempTotEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            tempTotVir[blockIdx.x * MAXDIM + threadIdx.x] += tempTotVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
        }
        reduce = reduce / 2;
        __syncthreads();
        //
    }
    __syncthreads();
    printf("total energy %f ", tempTotEn[blockIdx.x * MAXDIM]);
    
    return 0;
}

__device__ int double_mol_pair_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempTotVir, int fmol, int smol, float rcut){
    //fmol smol - indexes of firs and second molecules
    //get atom numbers
    float dmx;  //diffs in molecule coordinats
    float dmy;
    float dmz;
    float dax;  //diff in atoms coordinate
    float day;
    float daz;
    float r;    //distance ^2 of molecules
    float ra;   //distance ^2 of atoms
    int id;     //array indexes for potential parameters
    
    dmx = gDBox.xm[fmol] - gDBox.xm[smol];
    dmy = gDBox.ym[fmol] - gDBox.ym[smol];
    dmz = gDBox.zm[fmol] - gDBox.zm[smol];
    
    if(dmx > rcut){  //
        dmx = -rcut * 2.0 + dmx;
    }
    if(dmy > rcut ){
        dmy = -rcut * 2.0 + dmy;
    }
    if(dmz > rcut){
        dmz = -rcut * 2.0 + dmz;
    }
    if(dmx < -rcut){ //
        dmx = rcut * 2.0 + dmx;
    }
    if(dmy < -rcut){
        dmy = rcut*2.0 + dmy;
    }
    if(dmz < -rcut){
        dmz = rcut*2.0 + dmz;
    }
    r = dmx * dmx + dmy * dmy + dmz * dmz;
    
    if(r < rcut*rcut){ //if distance < rcut
        // for all atoms
        for(int i = 0; i < gTop.aNum[gDBox.mType[fmol]] ; i++){
            for(int j = 0; j < gTop.aNum[gDBox.mType[smol]]; j++){
                id = 1; // gBox.aType[gBox.fAtom[b] + j] * gConf.potNum[0] + gBox.aType[gBox.fAtom[a] + i];
                id = gTop.aType[i] * gConf.potNum[0] + gTop.aType[j];
                dax = gDBox.xa[gDBox.fAtomOfMol[fmol]+i] - gDBox.xa[gDBox.fAtomOfMol[smol] + j] + dmx;
                day = gDBox.ya[gDBox.fAtomOfMol[fmol]+i] - gDBox.ya[gDBox.fAtomOfMol[smol] + j] + dmy;
                daz = gDBox.za[gDBox.fAtomOfMol[fmol]+i] - gDBox.za[gDBox.fAtomOfMol[smol] + j] + dmy;
                
                ra = dax * dax + day * day + daz * daz;
                ra = gTop.sigma[id] * gTop.sigma[id] / ra;
                ra = ra * ra * ra;  //6 power
                //calculate potential
                tempTotEn[threadIdx.x] += gTop.epsi[id] * (ra* ra - ra);
                tempTotVir[threadIdx.x] += gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra);
            }
        }
    }
    
    return 0;
}

__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, float* &tempTotEn, float* &tempMolEn, float* &tempTotVir, float* &tempMolVir, int mol){
    
    return 1;
}


