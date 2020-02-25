#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>
#include "../mcrec.h"
//#include "double_trans.cu"

//__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int curDev);
//__device__  int double_totalen(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim);
//__device__ int double_mol_pair_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int fmol, int smol, float rcut);
//__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int mol);

//double_step 
__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int curDev);
__device__  int double_totalen(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim);
__device__ int double_mol_pair_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int fmol, int smol, float rcut);
__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int mol);

//double_trans
__device__ int double_transition(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, int curMol, curandState devStates);

//double volume change
__device__ int double_vol_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState devStates);

///////

int double_equilibration(gDoublebox* gDBox, hDoubleBox doubleBox, gOptions* gConf, gMolecula* gTop){
    cudaError_t cuErr;
    int* xDim;
    int temp;
    
    xDim = (int*) malloc(deviceCount * sizeof(int));
    for(int curDev = 0; curDev < deviceCount; curDev++){
        
        //set numbers of block equal to numbers of plate per device
        //numbers of thread equal to maximum numbers of molecules 
        temp = 0;
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){    //get maximum number of molecules in liquib phase
//            printf("liq %d plate %d molecules %d\n", curDev, doubleBox.platesPerDevice[curDev][i], doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]]);
            if(doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]] > temp){
                temp = doubleBox.nLiq[doubleBox.platesPerDevice[curDev][i]];
            }
        }
        for(int i = 0; i < doubleBox.devicePlates[curDev]; i++){    //get maximum numbers of molecules in vapor phase
//            printf("vap %d plate %d molecules %d\n", curDev, doubleBox.platesPerDevice[curDev][i], doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]]);
            if(doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]] > temp){
                temp = doubleBox.nVap[doubleBox.platesPerDevice[curDev][i]];
            }
        }
//        printf("device max %d\n",doubleBox.devicePlates[curDev]);
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
    printf("device count %d\n", deviceCount);
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //calculate equlibration
        printf("dev %d block %d thread %d\n", curDev, doubleBox.devicePlates[curDev], xDim[curDev]);
        double_equilib_cycle<<<doubleBox.devicePlates[curDev], xDim[curDev] >>>(gDBox[curDev], gConf[curDev], gTop[curDev], curDev);
        cuErr = cudaGetLastError();
        if(cuErr != cudaSuccess){
            printf("Error on device %d %s line %d, err: %s\n",curDev, __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //0, gDBox[curDev].stream
    }
    cudaDeviceSynchronize();    //sync after complit all equlibrations
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cudaStreamSynchronize(gDBox[curDev].stream);
        
    }
    return 0;
}

__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int curDev){
    int temp;
    int yDim;   //y Dimension of array
    float randTrans;    //random transition
    int randMol;    //random molecule
    curandState devStates;
    
    
    curand_init(1234, threadIdx.x, 0, &devStates);
    
    //check
    
    yDim = (gDBox.molNum[blockIdx.x] / blockDim.x) + 1; //set y dimension
    if(gDBox.molNum[blockIdx.x] > 0){
        if(threadIdx.x == 0){
            
            printf("molecules on device %d mol %d bdim %d ydim %d \n", curDev, gDBox.molNum[blockIdx.x], blockDim.x , yDim);
        }
        __syncthreads;  //sync yDim
        if(threadIdx.x == 0){   //shared for whjole block
    //        printf("device %d thread %d\n", curDev, threadIdx.x);
    //        printf("sigma %f %f %f %f, cuurent device %d\n", gTop.sigma[0], gTop.sigma[1], gTop.sigma[2], gTop.sigma[3], curDev );
    //        printf("ref en %f device %d nmol %d\n", gDBox.refEn[blockIdx.x], curDev, gDBox.molNum[blockIdx.x]);
    //        printf("mol id %d x %f y %f z %f \n", gDBox.fMolOnPlate[blockIdx.x], gDBox.xm[4], gDBox.ym[4], gDBox.zm[4]);
        }
    //    __syncthreads();
        double_totalen(gDBox, gConf, gTop, yDim);
    //    printf("block %d molecules %d\n", blockIdx.x, gDBox.molNum[blockIdx.x]);
        if(gDBox.molNum[blockIdx.x] > 0){ //cycle only if plate not empty
            
        
            for(int i = 0; i < 1000; i++){  //loop for some (set as a parameter to option file)
                randTrans = curand_uniform(&devStates); //get random transition
                if(randTrans < 0.5){    //move molecule
                    randMol = curand_uniform(&devStates) * gDBox.molNum[blockIdx.x] + gDBox.fMolOnPlate[blockIdx.x];
                    double_transition(gDBox, gConf, gTop, yDim, randMol, devStates);
                }
                else if(randTrans < 0.01){  //change volume
                    double_vol_change(gDBox, gConf, gTop, yDim, devStates);
                }
                else if(randTrans < 0.001){ //molecule transition
                    
                }
                //print properties
                if(threadIdx.x ==0){
                    printf("step %d liquid energy %f accept %d\n", i, gDBox.liqEn[blockIdx.x], gDBox.accLiqTrans[blockIdx.x]);
                }
            }
            //check equlibration status
        }
        else{
            //mark block as a compleet
        }
    }
    else{
        if(threadIdx.x == 0){
            printf("no molecules on block %d device %d\n", blockIdx.x, curDev);
        }
    }
}

__device__ int double_totalen(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim){
    int curMol; //current molecule
    int curMol2;    //second molecule
    int curId;  //current index
    int reduce;
    //nuber of plate equal block number 
//        if(threadIdx.x == 0){   //shared for whjole block
//         printf("test 13.02-2\n");
//        }
    //printf("start totalen %d\n",threadIdx.x);
    
    //======calculate liquid phase energy/virial
    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;  //set energy to zero
    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0; //set virial to zero
    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;  //set energy to zero
    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0; //set virial to zero
    //printf("done %d yDim %d\n", threadIdx.x, yDim);

    for(int i = 0; i < yDim; i++){
        curId = threadIdx.x * yDim + i; //current id of molecule in liqud list
        curMol = gDBox.fMolOnPlate[blockIdx.x] + gDBox.liqList[curId];  //get curent molecule GPU index
        if(gDBox.phaseType[curMol] == LIQ){  //calculate liquid energy
            //printf("cur mol %d mol liq %d \n", curId, gDBox.nLiq[blockIdx.x]);
            if(curId < gDBox.nLiq[blockIdx.x]){
                
                //set here intermolecular potential
                //printf("mol1 %d mol2 %d\n", curMol, curId);
                //calculate inermolecullar interaction
                for(int j = curId + 1; j < gDBox.nLiq[blockIdx.x]; j++){
                    curMol2 = gDBox.fMolOnPlate[blockIdx.x] + gDBox.liqList[j];
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
                }
            }
            else{
                //set energy to zero 
                gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] += 0.0;
                gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] += 0.0;
            }
        }
        else{   //calculate vapor energy
            if(curId < gDBox.nVap[blockIdx.x]){
                
                //set here intermolecular potential
                //printf("mol1 %d mol2 %d\n", curMol, curId);
                //calculate inermolecullar interaction
                for(int j = curId + 1; j < gDBox.nLiq[blockIdx.x]; j++){
                    curMol2 = gDBox.fMolOnPlate[blockIdx.x] + gDBox.liqList[j];
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
                }
            }
        }
    }
    //printf("%f ", gDBox.tempEn[blockIdx.x * MAXDIM + threadIdx.x]);
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    reduce = blockDim.x / 2;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
        }
        reduce = reduce / 2;
        __syncthreads();
        //
    }
    __syncthreads();
    if(threadIdx.x == 0){
        printf("device block %d liq molecules %d\n", blockIdx.x, gDBox.nLiq[blockIdx.x]);
        printf("plate %d total liquid energy %f vapor energy %f\n", blockIdx.x, gDBox.tempLiqEn[blockIdx.x * MAXDIM], gDBox.tempVapEn[blockIdx.x * MAXDIM]);
        gDBox.liqEn[blockIdx.x] = gDBox.tempLiqEn[blockIdx.x * MAXDIM];
        gDBox.vapEn[blockIdx.x] = gDBox.tempVapEn[blockIdx.x * MAXDIM];
        gDBox.liqVir[blockIdx.x] = gDBox.tempLiqVir[blockIdx.x * MAXDIM];
        gDBox.vapVir[blockIdx.x] = gDBox.tempVapVir[blockIdx.x * MAXDIM];
    }
    __syncthreads();
    return 0;
}

__device__ int double_mol_pair_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int fmol, int smol, float rcut){
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
        dmy = rcut * 2.0 + dmy;
    }
    if(dmz < -rcut){
        dmz = rcut * 2.0 + dmz;
    }
    r = dmx * dmx + dmy * dmy + dmz * dmz;
    //printf("i %d  j %d r %f rcut %f\n", fmol, smol, sqrt(r), rcut);
    if(r < rcut*rcut){ //if distance < rcut
        // for all atoms
        for(int i = 0; i < gTop.aNum[gDBox.mType[fmol]] ; i++){
            for(int j = 0; j < gTop.aNum[gDBox.mType[smol]]; j++){
                //id = 1; // gBox.aType[gBox.fAtom[b] + j] * gConf.potNum[0] + gBox.aType[gBox.fAtom[a] + i];
                id = gTop.aType[i] * gConf.potNum[0] + gTop.aType[j];
                dax = gDBox.xa[gDBox.fAtomOfMol[fmol]+i] - gDBox.xa[gDBox.fAtomOfMol[smol] + j] + dmx;
                day = gDBox.ya[gDBox.fAtomOfMol[fmol]+i] - gDBox.ya[gDBox.fAtomOfMol[smol] + j] + dmy;
                daz = gDBox.za[gDBox.fAtomOfMol[fmol]+i] - gDBox.za[gDBox.fAtomOfMol[smol] + j] + dmz;
                
                ra = dax * dax + day * day + daz * daz;
                //printf("x %f %f y %f %f z %f %f r %f ra %f \n", dmx, dax, dmy, day, dmz, daz, r, ra);
                //printf("mola %f mola %f\n", gDBox.za[gDBox.fAtomOfMol[fmol]+i], gDBox.za[gDBox.fAtomOfMol[smol] + j]);
                //printf("%d %d rm %f ra %f en %f vir %f\n",fmol, smol, sqrt(r), sqrt(ra), gTop.epsi[id] * (ra* ra - ra), gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra));
                ra = gTop.sigma[id] * gTop.sigma[id] / ra;
                ra = ra * ra * ra;  //6 power
                //calculate potential
                if(gDBox.phaseType[fmol] == LIQ){
                    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] += gTop.epsi[id] * (ra* ra - ra);
                    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] += gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra);
                }
                else{
                    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] += gTop.epsi[id] * (ra* ra - ra);
                    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] += gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra);
                }
                
            }
        }
        //printf("block %d thread %d en %f vir %f\n", blockIdx.x, threadIdx.x, gDBox.tempEn[blockIdx.x * MAXDIM + threadIdx.x], gDBox.tempEn[blockIdx.x * MAXDIM + threadIdx.x]);
    }
    
    return 0;
}

__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int mol){
    
    return 1;
}

__device__ int double_transition(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, int curMol, curandState devStates){
    //curMol -- gpu indexed molecule
    float oldEn;
    float newEn;
    float oldVir;
    float newVir;
    float oldX, oldY, oldZ; //old coordinates
    int curMol2;    //second molecule
    int reduce;
    
    //save old state------------------------------------------------------------
    if(threadIdx.x == 0){
        oldX = gDBox.xm[curMol];
        oldY = gDBox.ym[curMol];
        oldZ = gDBox.zm[curMol];
    }
    //zeros energy
    for(int i = 0; i < yDim; i++){
        gDBox.tempLiqEn[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempVapEn[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempLiqVir[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempVapVir[threadIdx.x * yDim + i] = 0.0;
    }
    
    //calculate old energy------------------------------------------------------
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if(gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]){
            if(gDBox.phaseType[curMol2] == LIQ){
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
            }
            else{
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
            }
        }
    }
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    reduce = blockDim.x / 2;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
        }
        reduce = reduce / 2;
        __syncthreads();
        //
    }
    if(threadIdx.x == 0){
        if(gDBox.phaseType[curMol] == LIQ){
            oldEn = gDBox.tempLiqEn[blockIdx.x * MAXDIM];
        }
        else{
            oldEn = gDBox.tempVapEn[blockIdx.x * MAXDIM];
        }
    }
    //generate new state--------------------------------------------------------
    if(threadIdx.x == 0){
        if(gDBox.phaseType[curMol] == LIQ){
            gDBox.xm[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
            gDBox.ym[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
            gDBox.zm[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
        }
        else{
            gDBox.xm[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
            gDBox.ym[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
            gDBox.zm[curMol] += (1.0 + curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
        }
    }
    __syncthreads();
    //calculate new energy------------------------------------------------------
    //zeros energy
    for(int i = 0; i < yDim; i++){
        gDBox.tempLiqEn[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempVapEn[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempLiqVir[threadIdx.x * yDim + i] = 0.0;
        gDBox.tempVapVir[threadIdx.x * yDim + i] = 0.0;
    }
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if(gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]){
            if(gDBox.phaseType[curMol2] == LIQ){
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
            }
            else{
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
            }
        }
    }
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    reduce = blockDim.x / 2;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x + reduce];
            gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] += gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x + reduce];
        }
        reduce = reduce / 2;
        __syncthreads();
        //
    }
    if(threadIdx.x == 0){
        if(gDBox.phaseType[curMol] == LIQ){
            newEn = gDBox.tempLiqEn[blockIdx.x * MAXDIM];
        }
        else{
            newEn = gDBox.tempVapEn[blockIdx.x * MAXDIM];
        }
    }
    __syncthreads;
    
    //check aceptance
    if(threadIdx.x == 0){
        if(curand_uniform(&devStates) < exp(-(newEn - oldEn)/gDBox.temp[blockIdx.x])){  //accept
            if(gDBox.phaseType[curMol] == LIQ){
                gDBox.accLiqTrans[blockIdx.x]++;
                gDBox.liqEn[blockIdx.x] += newEn - oldEn;
            }
            else{
                gDBox.accVapTrans[blockIdx.x]++;
                gDBox.vapEn[blockIdx.x] += newEn -oldEn;
            }
        }
        else{    //reject
            if(gDBox.phaseType[curMol] == LIQ){
                gDBox.rejLiqTrans[blockIdx.x]++;
            }
            else{
                gDBox.rejVapTrans[blockIdx.x]++;
            }
            //coordinates back
            gDBox.xm[curMol] = oldX;
            gDBox.ym[curMol] = oldY;
            gDBox.zm[curMol] = oldZ;
        }
    }
    __syncthreads;
    //calculate properties
    return 0;
}

__device__ int double_vol_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState devStates){
    //save state
    
    //calculate old enegry
    
    //change volumes
    
    //calculate new energy
    
    //check aceptance
    
    
    return 0;
}
