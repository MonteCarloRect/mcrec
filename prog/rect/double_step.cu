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
__device__ int double_transition(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates);

//double volume change
__device__ int double_vol_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates);

//double volume change
__device__ int double_liq_2_vap(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates);

//
__device__ int double_prop_calc(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim);

__device__ int double_prop_block_average(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, int curId);

__device__ int double_max_tran_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop);

__device__ int double_check_equilibration(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int curId);

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
            if(log2(temp) > 5){
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
    }
    
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cudaStreamSynchronize(gDBox[curDev].stream);
        
    }
    cudaDeviceSynchronize();    //sync after complit all equlibrations
    return 0;
}

__global__ void double_equilib_cycle(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int curDev){
    int temp;
    int yDim;   //y Dimension of array
    //float randTrans;    //random transition
    int randMol;    //random molecule
    curandState devStates;
    int curBlock;
    
    
    
    curand_init(1234, threadIdx.x+blockDim.x*blockIdx.x, 0, &devStates);
    //zeros all accept-reject
    gDBox.accLiqTrans[blockIdx.x] = 0;    //accept move in liquid phase
    gDBox.rejLiqTrans[blockIdx.x] = 0;
    gDBox.accVapTrans[blockIdx.x] = 0;   //accept move in vapor phase
    gDBox.rejVapTrans[blockIdx.x] = 0;
    gDBox.accVolChange[blockIdx.x] = 0;   //accept volume change
    gDBox.rejVolChange[blockIdx.x] = 0;
    gDBox.accLiq2Vap[blockIdx.x] = 0;    //accept liquid to vapor phase
    gDBox.rejLiq2Vap[blockIdx.x] = 0;
    gDBox.accVap2Liq[blockIdx.x] = 0;
    gDBox.rejVap2Liq[blockIdx.x] = 0;
    
    //check
    yDim = (gDBox.molNum[blockIdx.x] / blockDim.x) + 1; //set y dimension
    if(gDBox.molNum[blockIdx.x] > 0){
        if(threadIdx.x == 0){
            printf("molecules on device %d mol %d bdim %d ydim %d \n", curDev, gDBox.molNum[blockIdx.x], blockDim.x , yDim);
            curBlock = 0;
        }
        __syncthreads();  //sync yDim
//        if(threadIdx.x == 0){   //shared for whjole block
////            printf("device %d thread %d\n", curDev, threadIdx.x);
////            printf("sigma %f %f %f %f, cuurent device %d\n", gTop.sigma[0], gTop.sigma[1], gTop.sigma[2], gTop.sigma[3], curDev );
////            printf("ref en %f device %d nmol %d\n", gDBox.refEn[blockIdx.x], curDev, gDBox.molNum[blockIdx.x]);
////            printf("mol id %d x %f y %f z %f \n", gDBox.fMolOnPlate[blockIdx.x], gDBox.xm[4], gDBox.ym[4], gDBox.zm[4]);
//        }
    //    __syncthreads();
        double_totalen(gDBox, gConf, gTop, yDim);
        //printf("block %d molecules %d\n", blockIdx.x, gDBox.molNum[blockIdx.x]);
//        if(threadIdx.x == 0){
//            printf("energy 1 %f vir1 %f \n", gDBox.liqEn[blockIdx.x], gDBox.liqVir[blockIdx.x]);
//        }
//        double_totalen(gDBox, gConf, gTop, yDim);
        //printf("block %d molecules %d\n", blockIdx.x, gDBox.molNum[blockIdx.x]);
//        if(threadIdx.x == 0){
//            printf("energy 2 %f vir2 %f \n", gDBox.liqEn[blockIdx.x], gDBox.liqVir[blockIdx.x]);
//        }
        if(gDBox.molNum[blockIdx.x] > 0){ //cycle only if plate not empty
//            for(int i = 0; i < yDim; i++){
//                randMol = threadIdx.x + i + gDBox.fMolOnPlate[blockIdx.x];
//                if(randMol < gDBox.molNum[blockIdx.x]){
//                    printf("thread %d x %f y %f z %f\n", threadIdx.x, gDBox.xm[randMol], gDBox.ym[randMol], gDBox.zm[randMol]);
//                }
//                
//            }
            while(gDBox.eqStep[blockIdx.x] == 0){

            for(int nblk =0; nblk < EQBLOCKSIZE; nblk++){
            for(int step = 0; step < EQBLOCKCHECK; step++){  //loop for some (set as a parameter to option file)
                //randMol = curand_uniform(&devStates) * gDBox.molNum[blockIdx.x] + gDBox.fMolOnPlate[blockIdx.x];
                double_transition(gDBox, gConf, gTop, yDim, devStates);
                //double_totalen(gDBox, gConf, gTop, yDim);
                //if(threadIdx.x == 0){
                //    printf("energy 3 %f vir3 %f \n", gDBox.liqEn[blockIdx.x], gDBox.liqVir[blockIdx.x]);
                //}
                //print properties
//                if(step % 10000 == 0){
//                    if(threadIdx.x == 0){
//                        printf("step %d liquid energy %f vapor %f accept %d virial %f\n", step, gDBox.liqEn[blockIdx.x], gDBox.vapVir[blockIdx.x], gDBox.accLiqTrans[blockIdx.x], gDBox.liqVir[blockIdx.x]);
//                    }
//                    //check total energy
//                    double_totalen(gDBox, gConf, gTop, yDim);
//                    if(threadIdx.x == 0){
//                        printf("checked liquid en %f liqud vir %f\n", gDBox.liqEn[blockIdx.x], gDBox.liqVir[blockIdx.x]);
//                    }
//                }
                
                
            }   //end step loop
            if(nblk % 10 == 0){
                //printf("i %d second %d thread %d\n", nblk, blockIdx.x, threadIdx.x);
                double_vol_change(gDBox, gConf, gTop, yDim, devStates);
                double_liq_2_vap(gDBox, gConf, gTop, yDim, devStates);
            }
//            if(threadIdx.x == 0){
//                        printf("checked liquid en %f liqud vir %f\n", gDBox.liqEn[blockIdx.x], gDBox.liqVir[blockIdx.x]);
//                    }
            double_prop_calc(gDBox, gConf, gTop, yDim); //add to properties summ
            
            }   //end nblk loop
            double_prop_block_average(gDBox, gConf, gTop, yDim, curBlock);
            double_max_tran_change(gDBox, gConf, gTop);
            
            if(threadIdx.x == 0){
                curBlock++;
//                if(curBlock > 10){
//                    gDBox.eqStep[blockIdx.x] = curBlock;
//                }
            }
            __syncthreads();
            //check equlibration status
            double_check_equilibration(gDBox, gConf, gTop, curBlock);
            }   //end while 
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
    __shared__ int reduce;
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
    __syncthreads;

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
                for(int j = curId + 1; j < gDBox.nVap[blockIdx.x]; j++){
                    curMol2 = gDBox.fMolOnPlate[blockIdx.x] + gDBox.vapList[j];
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
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
//        printf("device block %d liq molecules %d\n", blockIdx.x, gDBox.nLiq[blockIdx.x]);
//        printf("plate %d total liquid energy %f vapor energy %f\n", blockIdx.x, gDBox.tempLiqEn[blockIdx.x * MAXDIM], gDBox.tempVapEn[blockIdx.x * MAXDIM]);
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
                dax = gDBox.xa[gDBox.fAtomOfMol[fmol] + i] - gDBox.xa[gDBox.fAtomOfMol[smol] + j] + dmx;
                day = gDBox.ya[gDBox.fAtomOfMol[fmol] + i] - gDBox.ya[gDBox.fAtomOfMol[smol] + j] + dmy;
                daz = gDBox.za[gDBox.fAtomOfMol[fmol] + i] - gDBox.za[gDBox.fAtomOfMol[smol] + j] + dmz;
                
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
        //printf("block %d thread %d en %f vir %f\n", blockIdx.x, threadIdx.x, gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x], gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x]);
    }
    
    return 0;
}

__device__ int double_mol_single_energy(gDoublebox gDBox, gOptions gConf, gMolecula gTop, int yDim, int mol){
    
    return 1;
}

__device__ int double_transition(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates){
    //curMol -- gpu indexed molecule
    __shared__ float oldEn;
    __shared__ float newEn;
    __shared__ float oldVir;
    __shared__ float newVir;
    __shared__ float3 sav; //old coordinates
    __shared__ int curMol;
    int curMol2;    //second molecule
    __shared__ int reduce;
    int id1;
    int id2;
    

    //save old state------------------------------------------------------------
    if(threadIdx.x == 0){
        curMol = curand_uniform(&devStates) * gDBox.molNum[blockIdx.x] + gDBox.fMolOnPlate[blockIdx.x];
        sav.x = gDBox.xm[curMol];
        sav.y = gDBox.ym[curMol];
        sav.z = gDBox.zm[curMol];
        oldEn = 0.0;
        newEn = 0.0;
        oldVir = 0.0;
        newVir = 0.0;
    }
//    //zeros energy
//    if(threadIdx.x == 0){
//        printf("part1\n");
//    }
    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    __syncthreads;
    //printf("thread %d curmol %d\n", threadIdx.x, curMol);
    //calculate old energy------------------------------------------------------
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if((gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]) && (curMol2 != curMol)){
            if(gDBox.phaseType[curMol2] == LIQ){
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
            }
            else{
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
            }
        }
    }
    //printf("thread %d curmol %d fmop %d en %f\n", threadIdx.x, curMol,gDBox.fMolOnPlate[blockIdx.x],gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x]);
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    //printf("thread %d energy %f\n", threadIdx.x, gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x]);
    reduce = blockDim.x / 2;
    __syncthreads;
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
            oldVir = gDBox.tempLiqVir[blockIdx.x * MAXDIM];
        }
        else{
            oldEn = gDBox.tempVapEn[blockIdx.x * MAXDIM];
            oldVir = gDBox.tempVapVir[blockIdx.x * MAXDIM];
        }
    }
    //generate new state--------------------------------------------------------
    if(threadIdx.x == 0){
        if(gDBox.phaseType[curMol] == LIQ){
            gDBox.xm[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
            gDBox.ym[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
            gDBox.zm[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxLiqTrans[blockIdx.x];
            //check out of boxes
            if(gDBox.xm[curMol] > 2.0 * gDBox.liqRcut[blockIdx.x]){
                gDBox.xm[curMol] -= 2.0 * gDBox.liqRcut[blockIdx.x];
            }
            if(gDBox.ym[curMol] > 2.0 * gDBox.liqRcut[blockIdx.x]){
                gDBox.ym[curMol] -= 2.0 * gDBox.liqRcut[blockIdx.x];
            }
            if(gDBox.zm[curMol] > 2.0 * gDBox.liqRcut[blockIdx.x]){
                gDBox.zm[curMol] -= 2.0 * gDBox.liqRcut[blockIdx.x];
            }
            if(gDBox.xm[curMol] < 0.0){
                gDBox.xm[curMol] += 2.0 * gDBox.liqRcut[blockIdx.x];
            }
            if(gDBox.ym[curMol] < 0.0){
                gDBox.ym[curMol] += 2.0 * gDBox.liqRcut[blockIdx.x];
            }
            if(gDBox.zm[curMol] < 0.0){
                gDBox.zm[curMol] += 2.0 * gDBox.liqRcut[blockIdx.x];
            }
        }
        else{
            gDBox.xm[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
            gDBox.ym[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
            gDBox.zm[curMol] += (1.0 - curand_uniform(&devStates)) * gDBox.maxVapTrans[blockIdx.x];
            //check out of boxes
            if(gDBox.xm[curMol] > 2.0 * gDBox.vapRcut[blockIdx.x]){
                gDBox.xm[curMol] -= 2.0 * gDBox.vapRcut[blockIdx.x];
            }
            if(gDBox.ym[curMol] > 2.0 * gDBox.vapRcut[blockIdx.x]){
                gDBox.ym[curMol] -= 2.0 * gDBox.vapRcut[blockIdx.x];
            }
            if(gDBox.zm[curMol] > 2.0 * gDBox.vapRcut[blockIdx.x]){
                gDBox.zm[curMol] -= 2.0 * gDBox.vapRcut[blockIdx.x];
            }
            if(gDBox.xm[curMol] < 0.0){
                gDBox.xm[curMol] += 2.0 * gDBox.vapRcut[blockIdx.x];
            }
            if(gDBox.ym[curMol] < 0.0){
                gDBox.ym[curMol] += 2.0 * gDBox.vapRcut[blockIdx.x];
            }
            if(gDBox.zm[curMol] < 0.0){
                gDBox.zm[curMol] += 2.0 * gDBox.vapRcut[blockIdx.x];
            }
        }
        
    }
    __syncthreads();
    //calculate new energy------------------------------------------------------
    //zeros energy
//    if(threadIdx.x == 0){
//        printf("part2\n");
//    }
    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    __syncthreads;
    
    //calculate new energy------------------------------------------------------
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if((gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]) && (curMol2 != curMol)){
            if(gDBox.phaseType[curMol2] == LIQ){
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
            }
            else{
                double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
            }
        }
    }
    //printf("thread %d curmol %d fmop %d en %f\n", threadIdx.x, curMol,gDBox.fMolOnPlate[blockIdx.x],gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x]);
    __syncthreads();    //chech all slots are calculated
    //summ all energyes
    //printf("thread %d energy %f\n", threadIdx.x, gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x]);
    reduce = blockDim.x / 2;
    __syncthreads;
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
            newVir = gDBox.tempLiqVir[blockIdx.x * MAXDIM];
        }
        else{
            newEn = gDBox.tempVapEn[blockIdx.x * MAXDIM];
            newVir = gDBox.tempVapVir[blockIdx.x * MAXDIM];
        }
    }
    __syncthreads;
    
    //check aceptance
    if(threadIdx.x == 0){
        
        if(curand_uniform(&devStates) < exp(-(newEn - oldEn)/gDBox.temp[blockIdx.x])){  //accept
            //printf("delta en %f %f curMol %d exp %f del en %f\n", oldEn, newEn, curMol, exp(-(newEn - oldEn)/gDBox.temp[blockIdx.x]), newEn - oldEn);
            if(gDBox.phaseType[curMol] == LIQ){
                gDBox.accLiqTrans[blockIdx.x]++;
                gDBox.liqEn[blockIdx.x] += newEn - oldEn;
                gDBox.liqVir[blockIdx.x] += newVir - oldVir;
            }
            else{
                gDBox.accVapTrans[blockIdx.x]++;
                gDBox.vapEn[blockIdx.x] += newEn - oldEn;
                gDBox.vapVir[blockIdx.x] += newVir - oldVir;
            }
            //rebuild lists

        }
        else{    //reject
            if(gDBox.phaseType[curMol] == LIQ){
                gDBox.rejLiqTrans[blockIdx.x]++;
                gDBox.phaseType[curMol] == VAP;
            }
            else{
                gDBox.rejVapTrans[blockIdx.x]++;
                gDBox.phaseType[blockIdx.x] = LIQ;
                
            }
            //coordinates back
            gDBox.xm[curMol] = sav.x;
            gDBox.ym[curMol] = sav.y;
            gDBox.zm[curMol] = sav.z;
            
        }
    }
    __syncthreads;
    //
    return 0;
}

__device__ int double_vol_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates){
    float oldLiqEn;
    float oldVapEn;
    float newLiqEn;
    float newVapEn;
    float oldLiqVir;
    float oldVapVir;
    int curMol;
    int curId;
    
    float savLiqRcut;
    float savVapRcut;
    float savLiqVol;
    float savVapVol;
    
    float dv;
    __shared__ float coef1;
    __shared__ float coef2;
    
    
    //save state
    for(int i = 0; i < yDim; i++){
        curId = threadIdx.x * yDim + i;
        curMol = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if(curMol < gDBox.molNum[blockIdx.x]){
            gDBox.tempXm[curId] = gDBox.xm[curMol];
            gDBox.tempYm[curId] = gDBox.ym[curMol];
            gDBox.tempZm[curId] = gDBox.zm[curMol];
        }
    }
    if(threadIdx.x == 0){
        savLiqRcut = gDBox.liqRcut[blockIdx.x];
        savVapRcut = gDBox.vapRcut[blockIdx.x];
        savLiqVol = gDBox.liqVol[blockIdx.x];
        savVapVol = gDBox.vapVol[blockIdx.x];
    }
    //printf("thread %d rcut %f %f vol %f %f\n", threadIdx.x, savLiqRcut, savVapRcut, savLiqVol, savVapVol);
    __syncthreads;
    //calculate old enegry  //current energy is old energy
    //double_totalen(gDBox, gConf, gTop, yDim);
    //__syncthreads;
    if(threadIdx.x == 0){
        oldLiqEn = gDBox.liqEn[blockIdx.x];
        oldVapEn = gDBox.vapEn[blockIdx.x];
        oldLiqVir = gDBox.liqVir[blockIdx.x];
        oldVapVir = gDBox.vapVir[blockIdx.x];
    }
    //printf("thread %d old energy %f\n",threadIdx.x, oldLiqEn);
    //change volumes
    if(threadIdx.x == 0){
        if(gDBox.liqVol[blockIdx.x] > gDBox.vapVol[blockIdx.x]){
            dv = gDBox.vapVol[blockIdx.x] * gDBox.maxVolChange[blockIdx.x] * (curand_uniform(&devStates) - 0.5);
        }
        else{
            dv = gDBox.vapVol[blockIdx.x] * gDBox.maxVolChange[blockIdx.x] * (curand_uniform(&devStates) - 0.5);
        }
        //printf("delta vol %f\n", dv);
        coef1 = pow((gDBox.liqVol[blockIdx.x] + dv) / gDBox.liqVol[blockIdx.x], 1.0/3.0);
        coef2 = pow((gDBox.vapVol[blockIdx.x] - dv) / gDBox.vapVol[blockIdx.x], 1.0/3.0);
    }
    __syncthreads;
//    //printf("thread %d coef1 %f, coef2 %f\n", threadIdx.x, coef1, coef2);
//    if(threadIdx.x == 0){   //VARIANT 1=========================================
//        for(int i = 0; i < gDBox.molNum[blockIdx.x]; i++){
//            curMol = i + gDBox.fMolOnPlate[blockIdx.x];
//            if(gDBox.phaseType[curMol] == LIQ){
//                gDBox.xm[curMol] *= coef1;
//                gDBox.ym[curMol] *= coef1;
//                gDBox.zm[curMol] *= coef1;
//            }
//            else{
//                gDBox.xm[curMol] *= coef2;
//                gDBox.ym[curMol] *= coef2;
//                gDBox.zm[curMol] *= coef2;
//            }
//            //printf("thread %d x %f y %f z %f\n", threadIdx.x, gDBox.xm[curMol], gDBox.ym[curMol], gDBox.zm[curMol]);
//        }
//    }
//    __syncthreads;
//    //VARIANT 2 ===============================================================
    for(int i = 0; i < yDim; i++){
        curMol = threadIdx.x + i + gDBox.fMolOnPlate[blockIdx.x];
        if(gDBox.phaseType[curMol] == LIQ){
            gDBox.xm[curMol] *= coef1;
            gDBox.ym[curMol] *= coef1;
            gDBox.zm[curMol] *= coef1;
        }
        else{
            gDBox.xm[curMol] *= coef2;
            gDBox.ym[curMol] *= coef2;
            gDBox.zm[curMol] *= coef2;
        }
//        printf("thread vol %d curMol %d x %f y %f z %f c1 %f c2 %f\n", threadIdx.x, curMol, gDBox.xm[curMol], gDBox.ym[curMol], gDBox.zm[curMol], coef1, coef2);
    }
    __syncthreads;
    //change volume and rcut
    if(threadIdx.x == 0){
        //printf("thread %d x %f y %f z %f\n", threadIdx.x, gDBox.xm[1], gDBox.ym[1], gDBox.zm[1]);
        gDBox.liqVol[blockIdx.x] += dv;
        gDBox.vapVol[blockIdx.x] -= dv;
        gDBox.liqRcut[blockIdx.x] = pow(gDBox.liqVol[blockIdx.x],1.0/3.0);
        gDBox.vapRcut[blockIdx.x] = pow(gDBox.vapVol[blockIdx.x],1.0/3.0);
        //printf("lv %f vv %f lr %f vr %f\n", gDBox.liqVol[blockIdx.x], gDBox.vapVol[blockIdx.x], gDBox.liqRcut[blockIdx.x], gDBox.vapRcut[blockIdx.x] );
    }
    __syncthreads;
    //calculate new energy
    double_totalen(gDBox, gConf, gTop, yDim);
    __syncthreads;
    if(threadIdx.x == 0){
        newLiqEn = gDBox.liqEn[blockIdx.x];
        newVapEn = gDBox.vapEn[blockIdx.x];
    }
    __syncthreads;
//    if(threadIdx.x == 100){
//        printf("volchange ver %f %f %f exp %f \n", dv, newLiqEn, oldLiqEn, exp(-(newLiqEn + newVapEn - oldLiqEn - oldVapEn)/gDBox.temp[blockIdx.x]) );
//    }
//    if(threadIdx.x == 0){
//        printf("Vol change dV %f nE %f oE %f exp %f \n", dv, newLiqEn, oldLiqEn, exp(-(newLiqEn + newVapEn - oldLiqEn - oldVapEn)/gDBox.temp[blockIdx.x] - gDBox.nLiq[blockIdx.x] * log(coef1 * coef1 * coef1) - gDBox.nVap[blockIdx.x] * log(coef2 * coef2 * coef2) ) );
//    }
    //check aceptance
    if(threadIdx.x == 0){
        if(curand_uniform(&devStates) < exp(-(newLiqEn + newVapEn - oldLiqEn - oldVapEn)/gDBox.temp[blockIdx.x] - gDBox.nLiq[blockIdx.x] * log(coef1 * coef1 * coef1) - gDBox.nVap[blockIdx.x] * log(coef2 * coef2 * coef2) ) ){  //accept
            //add to accept
            //printf("volume change accept\n");
            gDBox.accVolChange[blockIdx.x]++;
        }
        else{    //reject
            //coordinates back
            //printf("volume change reject s1 %f s2 %f\n", newLiqEn, oldVapEn);
            for(int i = 0; i < gDBox.molNum[blockIdx.x]; i++){
                //printf("i %d txm %f ym %f zm %f\n",i, gDBox.tempXm[i], gDBox.tempYm[i], gDBox.tempZm[i]);
                curMol = i + gDBox.fMolOnPlate[blockIdx.x];
                gDBox.xm[curMol] = gDBox.tempXm[i];
                gDBox.ym[curMol] = gDBox.tempYm[i];
                gDBox.zm[curMol] = gDBox.tempZm[i];
                //printf("i %d txm %f ym %f zm %f\n",i, gDBox.xm[curMol], gDBox.ym[curMol], gDBox.zm[curMol]);
            }
            gDBox.liqRcut[blockIdx.x] = savLiqRcut;
            gDBox.vapRcut[blockIdx.x] = savVapRcut;
            gDBox.liqVol[blockIdx.x] = savLiqVol;
            gDBox.vapVol[blockIdx.x] = savVapVol;
            gDBox.liqEn[blockIdx.x] = oldLiqEn;
            gDBox.vapEn[blockIdx.x] = oldVapEn;
            gDBox.liqVir[blockIdx.x] = oldLiqVir;
            gDBox.vapVir[blockIdx.x] = oldVapVir;
            gDBox.rejVolChange[blockIdx.x]++;
        }
    }
    __syncthreads;
    return 0;
}

__device__ int double_liq_2_vap(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, curandState &devStates){
    float newEn;
    float oldEn;
    float oldVir;
    float newVir;
    int curMol2;
    
    float oldx;
    float oldy;
    float oldz;
    __shared__ int curId;
    __shared__ int curMol;
    int reduce;
    int id1;
    int id2;
    
//    if(threadIdx.x < 50){
//        printf("test === thread %d\n", threadIdx.x);
//    }
    if(threadIdx.x == 0){
    //random molecule
        curId = curand_uniform(&devStates) * gDBox.molNum[blockIdx.x];
        curMol = curId + gDBox.fMolOnPlate[blockIdx.x];
        //save old state
        oldx = gDBox.xm[curMol];
        oldy = gDBox.ym[curMol];
        oldz = gDBox.zm[curMol];
        //printf("1 - old %f %f %f new %f %f %f\n", oldx, oldy, oldz, gDBox.xm[curMol], gDBox.ym[curMol], gDBox.zm[curMol]);
        //printf("==============thread %d curId %d curMol %d\n", threadIdx.x, curId, curMol);
    }
    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    __syncthreads;
//    
    //calculate old energy------------------------------------------------------
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if(curMol2 < gDBox.molNum[blockIdx.x]){
            if((gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]) && (curMol != curMol2)){
                if(gDBox.phaseType[curMol2] == LIQ){
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
                }
                else{
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
                }
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
            oldVir = gDBox.tempLiqVir[blockIdx.x * MAXDIM];
        }
        else{
            oldEn = gDBox.tempVapEn[blockIdx.x * MAXDIM];
            oldVir = gDBox.tempVapVir[blockIdx.x * MAXDIM];
        }
    }
    //random new position 
    if(threadIdx.x == 0){
        if(gDBox.phaseType[curMol] == LIQ){
            gDBox.xm[curMol] = curand_uniform(&devStates) * gDBox.vapRcut[blockIdx.x] * 2.0;
            gDBox.ym[curMol] = curand_uniform(&devStates) * gDBox.vapRcut[blockIdx.x] * 2.0;
            gDBox.zm[curMol] = curand_uniform(&devStates) * gDBox.vapRcut[blockIdx.x] * 2.0;
            gDBox.phaseType[curMol] = VAP;
            gDBox.nLiq[blockIdx.x]--;
            gDBox.nVap[blockIdx.x]++;
        }
        else{
            gDBox.xm[curMol] = curand_uniform(&devStates) * gDBox.liqRcut[blockIdx.x] * 2.0;
            gDBox.ym[curMol] = curand_uniform(&devStates) * gDBox.liqRcut[blockIdx.x] * 2.0;
            gDBox.zm[curMol] = curand_uniform(&devStates) * gDBox.liqRcut[blockIdx.x] * 2.0;
            gDBox.phaseType[curMol] = LIQ;
            gDBox.nLiq[blockIdx.x]++;
            gDBox.nVap[blockIdx.x]--;
        }
    }
    __syncthreads;
    //calculate new energy
    gDBox.tempLiqEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapEn[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempLiqVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    gDBox.tempVapVir[blockIdx.x * MAXDIM + threadIdx.x] = 0.0;
    //calculate new energy------------------------------------------------------
    for(int i = 0; i < yDim; i++){
        curMol2 = threadIdx.x * yDim + i + gDBox.fMolOnPlate[blockIdx.x];
        if(curMol2 < gDBox.molNum[blockIdx.x]){
            if((gDBox.phaseType[curMol2] == gDBox.phaseType[curMol]) && (curMol != curMol2)){
                if(gDBox.phaseType[curMol2] == LIQ){
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.liqRcut[blockIdx.x]);
                }
                else{
                    double_mol_pair_energy(gDBox, gConf, gTop, yDim, curMol, curMol2, gDBox.vapRcut[blockIdx.x]);
                }
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
    //check acceptance
    if(threadIdx.x == 0){
        //printf("2 - old %f %f %f new %f %f %f\n", oldx, oldy, oldz, gDBox.xm[curMol], gDBox.ym[curMol], gDBox.zm[curMol]);
        //printf("trans curmol %d oldEn %f newEn %f oldVir %f newVir %f \n", curMol, oldEn, newEn, oldVir, newVir);
        if(gDBox.phaseType[curMol] == LIQ){ //swiched to liquid
            if(curand_uniform(&devStates) < exp(-(newEn - oldEn)/gDBox.temp[blockIdx.x] + log(gDBox.vapVol[blockIdx.x] * (gDBox.nLiq[blockIdx.x]+1) / gDBox.liqVol[blockIdx.x] / gDBox.nVap[blockIdx.x]) ) ){   //accept
                //printf("vap2liq molecule transition accept\n");
                gDBox.vapEn[blockIdx.x] -= oldEn;
                gDBox.vapVir[blockIdx.x] -= oldVir;
                gDBox.liqEn[blockIdx.x] += newEn;
                gDBox.liqVir[blockIdx.x] += newVir;
                gDBox.accVap2Liq[blockIdx.x]++;
                id1 = 0;
                id2 = 0;
                for(int i = 0; i < gDBox.molNum[blockIdx.x]; i++){
                    if(gDBox.phaseType[gDBox.fMolOnPlate[blockIdx.x] + i] == LIQ){
                        gDBox.liqList[gDBox.fMolOnPlate[blockIdx.x] + id1] = i;
                        id1++;
                    }
                    else{
                        gDBox.vapList[gDBox.fMolOnPlate[blockIdx.x] + id2] = i;
                        id2++;
                    }
                }
            }
            else{   //reject
                //printf("vap2liq molecule transition reject\n");
                gDBox.xm[curMol] = oldx;
                gDBox.ym[curMol] = oldy;
                gDBox.zm[curMol] = oldz;
                gDBox.phaseType[curMol] = VAP;
                gDBox.nLiq[blockIdx.x]--;
                gDBox.nVap[blockIdx.x]++;
                gDBox.rejVap2Liq[blockIdx.x]++;
            }
        }
        else{   //switched to vapor
            if(curand_uniform(&devStates) < exp(-(newEn - oldEn)/gDBox.temp[blockIdx.x] + log(gDBox.liqVol[blockIdx.x] * (gDBox.nVap[blockIdx.x]+1) / gDBox.vapVol[blockIdx.x] / gDBox.nLiq[blockIdx.x]) ) ){   //accept
                //printf("liq2vap molecule transition accept\n");
                gDBox.liqEn[blockIdx.x] += newEn;
                gDBox.liqVir[blockIdx.x] += newVir;
                gDBox.vapEn[blockIdx.x] -= oldEn;
                gDBox.vapVir[blockIdx.x] -= oldVir;
                gDBox.accLiq2Vap[blockIdx.x]++;
                id1 = 0;
                id2 = 0;
                for(int i = 0; i < gDBox.molNum[blockIdx.x]; i++){
                    if(gDBox.phaseType[gDBox.fMolOnPlate[blockIdx.x] + i] == LIQ){
                        gDBox.liqList[gDBox.fMolOnPlate[blockIdx.x] + id1] = i;
                        id1++;
                    }
                    else{
                        gDBox.vapList[gDBox.fMolOnPlate[blockIdx.x] + id2] = i;
                        id2++;
                    }
                }
            }
            else{   //reject
                //printf("liq2vap molecule transition reject\n");
                gDBox.xm[curMol] = oldx;
                gDBox.ym[curMol] = oldy;
                gDBox.zm[curMol] = oldz;
                gDBox.phaseType[curMol] = LIQ;
                gDBox.nLiq[blockIdx.x]++;
                gDBox.nVap[blockIdx.x]--;
                gDBox.rejLiq2Vap[blockIdx.x]++;
            }
        }
    }
    __syncthreads;
    //printf("thread %d curMol %d\n", threadIdx.x, reduce);
    return 0;
}

__device__ int double_prop_calc(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim){

    if(threadIdx.x == 0){
        //energy
        gDBox.sumLiqEn[blockIdx.x] += gDBox.liqEn[blockIdx.x];
        gDBox.sumVapEn[blockIdx.x] += gDBox.vapEn[blockIdx.x];
        //molecule energy
        gDBox.sumLiqMolEn[blockIdx.x] += gDBox.nLiq[blockIdx.x];
        gDBox.sumVapMolEn[blockIdx.x] += gDBox.nVap[blockIdx.x];
        //molecule numbers by type
        for(int i = 0; i < gDBox.nLiq[blockIdx.x]; i++){
            gDBox.sumLiqMol[blockIdx.x * gConf.subNum[0] + gDBox.mType[gDBox.liqList[gDBox.fMolOnPlate[blockIdx.x] + i]]] += 1.0;
            gDBox.sumLiqMassDens[blockIdx.x] += 1.0 / gDBox.liqVol[blockIdx.x]; //now number density swich to mass
        }
        for(int i = 0; i < gDBox.nVap[blockIdx.x]; i++){
            gDBox.sumVapMol[blockIdx.x * gConf.subNum[0] + gDBox.mType[gDBox.vapList[gDBox.fMolOnPlate[blockIdx.x] + i]]] += 1.0;
            gDBox.sumVapMassDens[blockIdx.x] += 1.0 / gDBox.vapVol[blockIdx.x];
        } 
        for(int i = 0; i < gConf.subNum[0]; i++){
            if(gDBox.nLiq[blockIdx.x] > 0){
                gDBox.sumLiqConc[blockIdx.x * gConf.subNum[0] + i] = gDBox.sumLiqMol[blockIdx.x + i] / gDBox.nLiq[blockIdx.x];
            }
            if(gDBox.nVap[blockIdx.x] > 0){
                gDBox.sumVapConc[blockIdx.x * gConf.subNum[0] + i] = gDBox.sumVapMol[blockIdx.x + i] / gDBox.nVap[blockIdx.x];
            }
            
        }
        //pressure p = nkT + 1/3 W 
        gDBox.sumLiqPress[blockIdx.x] += gDBox.temp[blockIdx.x] * gDBox.nLiq[blockIdx.x] / gDBox.liqVol[blockIdx.x] * 1.38064852*10.0 - gDBox.liqVir[blockIdx.x] / gDBox.liqVol[blockIdx.x] / 3.0f * 1.38064852*10.0;
        gDBox.sumVapPress[blockIdx.x] += gDBox.temp[blockIdx.x] * gDBox.nVap[blockIdx.x] / gDBox.vapVol[blockIdx.x] * 1.38064852*10.0 - gDBox.vapVir[blockIdx.x] / gDBox.vapVol[blockIdx.x] / 3.0f * 1.38064852*10.0;
        //
        
    }
    
    __syncthreads();
    return 0;
}

__device__ int double_prop_block_average(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int yDim, int curId){
    
    if(threadIdx.x == 0){
        if(curId > EQBLOCKS-1){
            //swith up
            for(int i = 0; i < EQBLOCKS-1; i++){
                gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + i] = gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockVapEn[blockIdx.x * EQBLOCKS + i] = gDBox.blockVapEn[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockLiqMolEn[blockIdx.x * EQBLOCKS + i] = gDBox.blockLiqMolEn[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockVapMolEn[blockIdx.x * EQBLOCKS + i] = gDBox.blockVapMolEn[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockLiqPress[blockIdx.x * EQBLOCKS + i] = gDBox.blockLiqPress[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockVapPress[blockIdx.x * EQBLOCKS + i] = gDBox.blockVapPress[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + i] = gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + i + 1];
                gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + i] = gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + i + 1];
                for(int j = 0; j < gConf.subNum[0]; j++){
                    gDBox.blockLiqConc[blockIdx.x * EQBLOCKS + i * gConf.subNum[0] + j] = gDBox.blockLiqConc[blockIdx.x * EQBLOCKS + (i + 1) * gConf.subNum[0] + j];
                    gDBox.blockVapConc[blockIdx.x * EQBLOCKS + i * gConf.subNum[0] + j] = gDBox.blockVapConc[blockIdx.x * EQBLOCKS + (i + 1) * gConf.subNum[0] + j];
                }
                curId = EQBLOCKS - 1;
            }
        }
        //average
        gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + curId] = gDBox.sumLiqEn[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockVapEn[blockIdx.x * EQBLOCKS + curId] = gDBox.sumVapEn[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockLiqMolEn[blockIdx.x * EQBLOCKS + curId] = gDBox.sumLiqMolEn[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockVapMolEn[blockIdx.x * EQBLOCKS + curId] = gDBox.sumVapMolEn[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockLiqPress[blockIdx.x * EQBLOCKS + curId] = gDBox.sumLiqPress[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockVapPress[blockIdx.x * EQBLOCKS + curId] = gDBox.sumVapPress[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + curId] = gDBox.sumLiqMassDens[blockIdx.x] / (float) EQBLOCKSIZE;
        gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + curId] = gDBox.sumVapMassDens[blockIdx.x] / (float) EQBLOCKSIZE;
        for(int i = 0; i < gConf.subNum[0]; i++){
            gDBox.blockLiqConc[blockIdx.x * EQBLOCKS + curId * gConf.subNum[0] + i] = gDBox.sumLiqConc[blockIdx.x + i] / (float) EQBLOCKSIZE;
            gDBox.blockVapConc[blockIdx.x * EQBLOCKS + curId * gConf.subNum[0] + i] = gDBox.sumVapConc[blockIdx.x + i] / (float) EQBLOCKSIZE;
        }
        
        
        //zeros block
        gDBox.sumLiqEn[blockIdx.x] = 0.0;
        gDBox.sumVapEn[blockIdx.x] = 0.0;
        gDBox.sumLiqMolEn[blockIdx.x] = 0.0;
        gDBox.sumVapMolEn[blockIdx.x] = 0.0;
        gDBox.sumLiqPress[blockIdx.x] = 0.0;
        gDBox.sumVapPress[blockIdx.x] = 0.0;
        gDBox.sumLiqMassDens[blockIdx.x] = 0.0;
        gDBox.sumVapMassDens[blockIdx.x] = 0.0;
        for(int i = 0; i < gConf.subNum[0]; i++){
            gDBox.sumLiqMol[blockIdx.x * gConf.subNum[0] + i] = 0.0;
            gDBox.sumVapMol[blockIdx.x * gConf.subNum[0] + i] = 0.0;
            gDBox.sumLiqConc[blockIdx.x * gConf.subNum[0] + i] = 0.0;
            gDBox.sumVapConc[blockIdx.x * gConf.subNum[0] + i] = 0.0;
        }
        
         //PRINT propertyes
        printf("---------------------------------------------------------------\n");
        for(int curBlock = 0; curBlock < EQBLOCKS; curBlock++){
            printf("dev block %d %d liq en %f vap en\n", blockIdx.x, curBlock, gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + curBlock], curBlock, gDBox.blockVapEn[blockIdx.x * EQBLOCKS + curBlock]);
        }
        
        printf("liq en %f liq press %f vap en %f vap press %f\n", gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + curId], gDBox.blockLiqPress[blockIdx.x * EQBLOCKS + curId], gDBox.blockVapEn[blockIdx.x * EQBLOCKS + curId], gDBox.blockVapPress[blockIdx.x * EQBLOCKS + curId] );
        for(int i = 0; i < gConf.subNum[0]; i++){
            printf("sub %d of subnum %d liq conc %d %f vap conc %d %f\n",i, gConf.subNum[0], gDBox.blockLiqConc[blockIdx.x * gConf.subNum[0] + i], gDBox.blockLiqMol[blockIdx.x * gConf.subNum[0] + i], gDBox.blockVapConc[blockIdx.x * gConf.subNum[0] + i], gDBox.blockVapMol[blockIdx.x * gConf.subNum[0]]);
        }
        printf("dens liq %f vap %f\n", gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + curId], gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + curId]);
        printf("liq trans accept %d reject %d vap trans accept %d reject %d  \n", gDBox.accLiqTrans[blockIdx.x], gDBox.rejLiqTrans[blockIdx.x], gDBox.accVapTrans[blockIdx.x], gDBox.rejVapTrans[blockIdx.x]);
        printf("nliq %d nvap %d\n", gDBox.nLiq[blockIdx.x], gDBox.nVap[blockIdx.x]);
        printf("vol change accept %d reject %d\n", gDBox.accVolChange[blockIdx.x], gDBox.rejVolChange[blockIdx.x]);
        printf("trans move accept %d %d reject %d %d\n", gDBox.accLiq2Vap[blockIdx.x], gDBox.accVap2Liq[blockIdx.x], gDBox.rejLiq2Vap[blockIdx.x], gDBox.rejVap2Liq[blockIdx.x]);
    }
   
    
    __syncthreads();
    return 0;
}

__device__ int double_max_tran_change(gDoublebox &gDBox, gOptions gConf, gMolecula gTop){
    
    if(threadIdx.x == 0){
    //liqud
        if((gDBox.accLiqTrans[blockIdx.x]+1)/(gDBox.rejLiqTrans[blockIdx.x]+1) > 0.6){
            gDBox.maxLiqTrans[blockIdx.x]*=1.2;
        }
        if(gDBox.maxLiqTrans[blockIdx.x] > gDBox.liqRcut[blockIdx.x]){
            gDBox.maxLiqTrans[blockIdx.x] = gDBox.liqRcut[blockIdx.x];
        }
        if((gDBox.accLiqTrans[blockIdx.x]+1)/(gDBox.rejLiqTrans[blockIdx.x]+1) < 0.4){
            gDBox.maxLiqTrans[blockIdx.x]*=0.8;
        }
        if(gDBox.maxLiqTrans[blockIdx.x] < 0.01){
            gDBox.maxLiqTrans[blockIdx.x] < 0.01;
        }
    //vapor
        if((gDBox.accVapTrans[blockIdx.x]+1)/(gDBox.rejVapTrans[blockIdx.x]+1) > 0.6){
            gDBox.maxVapTrans[blockIdx.x]*=1.2;
        }
        if(gDBox.maxVapTrans[blockIdx.x] > gDBox.vapRcut[blockIdx.x]){
            gDBox.maxVapTrans[blockIdx.x] = gDBox.vapRcut[blockIdx.x];
        }
        if((gDBox.accVapTrans[blockIdx.x]+1)/(gDBox.rejVapTrans[blockIdx.x]+1) < 0.4){
            gDBox.maxLiqTrans[blockIdx.x]*=0.8;
        }
        if(gDBox.maxLiqTrans[blockIdx.x] < 0.01){
            gDBox.maxLiqTrans[blockIdx.x] < 0.01;
        }
    //vol change
        if((gDBox.accVolChange[blockIdx.x]+1)/(gDBox.rejVolChange[blockIdx.x]+1) > 0.5){
            gDBox.maxVolChange[blockIdx.x] *= 0.8;
        }
        if(gDBox.accVolChange[blockIdx.x] < 1){
            gDBox.maxVolChange[blockIdx.x] *= 1.1;
        }
    //zeroes accept/reject
        gDBox.accLiqTrans[blockIdx.x] = 0;
        gDBox.rejLiqTrans[blockIdx.x] = 0;
        gDBox.accVapTrans[blockIdx.x] = 0;
        gDBox.rejVapTrans[blockIdx.x] = 0;
        gDBox.accVolChange[blockIdx.x] = 0;
        gDBox.rejVolChange[blockIdx.x] = 0;
        gDBox.accLiq2Vap[blockIdx.x] = 0;
        gDBox.rejLiq2Vap[blockIdx.x] = 0;
        gDBox.accVap2Liq[blockIdx.x] = 0;
        gDBox.rejVap2Liq[blockIdx.x] = 0;
        
    }
    __syncthreads;
    return 0;
}

__device__ int double_check_equilibration(gDoublebox &gDBox, gOptions gConf, gMolecula gTop, int curId){
    float maxEn;
    float maxDens;
    float maxPres;
    
    if((threadIdx.x == 0) && (curId > EQBLOCKS)){
        //get average value
        gDBox.avLiqEn[blockIdx.x] = 0.0;
        gDBox.avVapEn[blockIdx.x] = 0.0;
        gDBox.avLiqPress[blockIdx.x] = 0.0;
        gDBox.avVapPress[blockIdx.x] = 0.0;
        gDBox.avLiqMol[blockIdx.x] = 0.0;
        gDBox.avVapMol[blockIdx.x] = 0.0;
        gDBox.avLiqMassDens[blockIdx.x] = 0.0;
        gDBox.avVapMassDens[blockIdx.x] = 0.0;
        for(int i = 0; i < gConf.subNum[0]; i++){
            gDBox.avLiqConc[blockIdx.x * gConf.subNum[0] + i] = 0.0;
            gDBox.avVapConc[blockIdx.x * gConf.subNum[0] + i] = 0.0;
        }
        for(int i = 0; i < EQBLOCKS; i++){
            gDBox.avLiqEn[blockIdx.x] += gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + i];
            gDBox.avVapEn[blockIdx.x] += gDBox.blockVapEn[blockIdx.x * EQBLOCKS + i];
            gDBox.avLiqPress[blockIdx.x] += gDBox.blockLiqPress[blockIdx.x * EQBLOCKS + i];
            gDBox.avVapPress[blockIdx.x] += gDBox.blockVapPress[blockIdx.x * EQBLOCKS + i];
            gDBox.avLiqMassDens[blockIdx.x] += gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + i];
            gDBox.avVapMassDens[blockIdx.x] += gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + i];
        }
        gDBox.avLiqEn[blockIdx.x] /= EQBLOCKS;
        gDBox.avVapEn[blockIdx.x] /= EQBLOCKS;
        gDBox.avLiqPress[blockIdx.x] /= EQBLOCKS;
        gDBox.avVapPress[blockIdx.x] /= EQBLOCKS;
        gDBox.avLiqMassDens[blockIdx.x] /= EQBLOCKS;
        gDBox.avVapMassDens[blockIdx.x] /= EQBLOCKS;
        
        maxEn = 0.0;
        maxPres = 0.0;
        maxPres = 0.0;
        //check energy
        for(int i = 0; i < EQBLOCKS; i++){
            if(maxEn < abs(gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + i] - gDBox.avLiqEn[blockIdx.x]) / abs(gDBox.avLiqEn[blockIdx.x] )){
                maxEn = abs(gDBox.blockLiqEn[blockIdx.x * EQBLOCKS + i] - gDBox.avLiqEn[blockIdx.x]) / abs(gDBox.avLiqEn[blockIdx.x]);
            }
            if(maxEn < abs(gDBox.blockVapEn[blockIdx.x * EQBLOCKS + i] - gDBox.avVapEn[blockIdx.x]) / abs(gDBox.avVapEn[blockIdx.x] )){
                maxEn = abs(gDBox.blockVapEn[blockIdx.x * EQBLOCKS + i] - gDBox.avVapEn[blockIdx.x]) / abs(gDBox.avVapEn[blockIdx.x]);
            }
            if(maxDens < abs(gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + i] - gDBox.avLiqMassDens[blockIdx.x]) / abs(gDBox.avLiqMassDens[blockIdx.x] )){
                maxDens = abs(gDBox.blockLiqMassDens[blockIdx.x * EQBLOCKS + i] - gDBox.avLiqMassDens[blockIdx.x]) / abs(gDBox.avLiqMassDens[blockIdx.x]);
            }
            if(maxDens < abs(gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + i] - gDBox.avVapMassDens[blockIdx.x]) / abs(gDBox.avVapMassDens[blockIdx.x] )){
                maxDens = abs(gDBox.blockVapMassDens[blockIdx.x * EQBLOCKS + i] - gDBox.avVapMassDens[blockIdx.x]) / abs(gDBox.avVapMassDens[blockIdx.x]);
            }
            if(maxPres < abs(gDBox.blockVapPress[blockIdx.x * EQBLOCKS + i] - gDBox.avVapPress[blockIdx.x]) / abs(gDBox.avVapPress[blockIdx.x] )){
                maxPres = abs(gDBox.blockVapPress[blockIdx.x * EQBLOCKS + i] - gDBox.avVapPress[blockIdx.x]) / abs(gDBox.avVapPress[blockIdx.x]);
            }
        }
        if(maxEn > 0.05){
            printf("====energy not equlibrated %f\n", maxEn);
        }
        if(maxDens > 0.05){
            printf("====density not equlibrated %f\n", maxDens);
        }
        if(maxPres > 0.05){
            printf("====pressure not equlibrated %f\n", maxPres);
        }
        if((maxEn < 0.05) && (maxDens < 0.05) && (maxPres < 0.05)){
            gDBox.eqStep[blockIdx.x] = curId;
        }
    }
    else{
        if(threadIdx.x == 0){
            printf("current block %d less %d\n", curId, EQBLOCKS);
        }
    }
    
    return 0;
}

