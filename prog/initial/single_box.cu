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
__device__ int single_get_prop(int yDim, gOptions gConf, gMolecula gTop, gSingleBox &gBox, int curId);
__device__ int single_change_trans(gOptions gConf, gMolecula gTop, gSingleBox &gBox);

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
    __shared__ float3 oldMol;
    __shared__ float3 oldState[MAXATOM];
    float deltaE;
    bool equlibrated;
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
    
    //EQULIBRATION
    equlibrated = false;
    while (equlibrated != true){
        for(int ncheck = 0; ncheck < EQBLOCKSIZE; ncheck++){
        for(int step= 0 ; step < EQBLOCKCHECK; step++){
            //get random numbers
            if(threadIdx.x==0){
                //gBox.curMol[blockIdx.x] = (curand_uniform(&devStates)*gBox.molNum[blockIdx.x]);
                randMol = curand_uniform(&devStates)*gBox.molNum[blockIdx.x] + gBox.fMol[blockIdx.x]; //number of molecule in array
                //printf("random %d %d  \n", i, randMol);
            }
            __syncthreads();
            //-------calculate old energy
            single_one_energy(yDim, randMol, gConf, gTop, gBox);
            if(threadIdx.x==0){ //get old energy
                gBox.oldEnergy[blockIdx.x] = gBox.mEnergy[gBox.fMol[blockIdx.x]]; //energy in zero molecule
                gBox.oldVirial[blockIdx.x] = gBox.mVirial[gBox.fMol[blockIdx.x]]; //virial i zero molecule
                //printf("old enegry %f virial %f \n", gBox.oldEnergy[blockIdx.x], gBox.oldVirial[blockIdx.x]);
                oldMol.x = gBox.xm[randMol];
                oldMol.y = gBox.ym[randMol];
                oldMol.z = gBox.zm[randMol];
                for(int i = 0; i < gTop.aNum[gBox.mType[randMol]]; i++){
                    oldState[i].x = gBox.xa[gBox.fAtom[randMol] + i];
                    oldState[i].y = gBox.ya[gBox.fAtom[randMol] + i];
                    oldState[i].z = gBox.za[gBox.fAtom[randMol] + i];
                }
            }
            __syncthreads();
            //------------change molecule configuration
            if(threadIdx.x==0){
                single_conf_change(randMol, gConf, gTop, gBox, devStates);
            }
            __syncthreads();
            //-------calculate new energy
            single_one_energy(yDim, randMol, gConf, gTop, gBox);
            if(threadIdx.x==0){ //get new energy
                gBox.newEnergy[blockIdx.x] = gBox.mEnergy[gBox.fMol[blockIdx.x]]; //energy in zero molecule
                gBox.newVirial[blockIdx.x] = gBox.mVirial[gBox.fMol[blockIdx.x]]; //virial i zero molecule
                //
            }
            //check aceptance
            if(threadIdx.x==0){
                deltaE=gBox.newEnergy[blockIdx.x] - gBox.oldEnergy[blockIdx.x];
                if(curand_uniform(&devStates) < exp(-gConf.Temp[blockIdx.x]*deltaE)){
//                printf("old enegry %f virial %f ", gBox.oldEnergy[blockIdx.x], gBox.oldVirial[blockIdx.x]);
//                printf("new enegry %f virial %f ", gBox.newEnergy[blockIdx.x], gBox.newVirial[blockIdx.x]);
//                printf(" dE %f w %f \n", deltaE, exp(-gConf.Temp[blockIdx.x]*deltaE));
                    //accept
                    //printf("accept\n");
                    gBox.accept[blockIdx.x]++;
                    //change energy
                    gBox.energy[blockIdx.x]+=gBox.newEnergy[blockIdx.x] - gBox.oldEnergy[blockIdx.x];
                    gBox.virial[blockIdx.x]+=gBox.newVirial[blockIdx.x] - gBox.oldVirial[blockIdx.x];
                }
                else{
                    //printf("reject\n");
                    gBox.reject[blockIdx.x]++;
                    //reject
                    //get old ccordinats
                    gBox.xm[randMol] = oldMol.x;
                    gBox.ym[randMol] = oldMol.y;
                    gBox.zm[randMol] = oldMol.z;
                    for(int i = 0; i < gTop.aNum[gBox.mType[randMol]]; i++){
                        gBox.xa[gBox.fAtom[randMol] + i] = oldState[i].x;
                        gBox.ya[gBox.fAtom[randMol] + i] = oldState[i].y;
                        gBox.za[gBox.fAtom[randMol] + i] = oldState[i].z;
                    }
                }
            }
            __syncthreads();
            
            
            //check maximumus
            if( (ncheck % 500 == 0) && (step==0) ){
                if(threadIdx.x==0){
                    printf("accept/reject %d / %d ", gBox.accept[blockIdx.x], gBox.reject[blockIdx.x]);
                }
                single_change_trans(gConf, gTop, gBox);
                if(threadIdx.x==0){
                    printf(" thr %d step %d energy %f virial %f max trans %f\n",threadIdx.x, ncheck, gBox.energy[blockIdx.x], gBox.virial[blockIdx.x], gBox.transMaxMove[blockIdx.x] );
                }
                single_calc_totenergy(yDim, gConf, gTop, gBox);
            }
            
        }
        //get properties
        single_get_prop(yDim, gConf, gTop, gBox, ncheck);
        
        }
        //check equlibration
        
        equlibrated = true;
    }
    if(threadIdx.x==0){
        printf("energy %f virial %f\n", gBox.energy[blockIdx.x], gBox.virial[blockIdx.x]);
        printf("accepted %d rejected %d\n", gBox.accept[blockIdx.x], gBox.reject[blockIdx.x]);
    }
    single_calc_totenergy(yDim, gConf, gTop, gBox);
}

__device__ int single_calc_totenergy(int yDim, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    int curMol;
    int curMol2;
    __shared__ int maxMol;
    float sumE;
    float sumV;
    __shared__ int reduce;
    maxMol = gBox.fMol[blockIdx.x]+blockDim.x*yDim;
    //printf("1 mol %d  lastmol %d\n", gBox.fMol[blockIdx.x], maxMol);
    for(int i = 0; i < yDim; i++){    //for several molecules per thread
        curMol=gBox.fMol[blockIdx.x]+i*blockDim.x+threadIdx.x;    //current number of molecule
//        gBox.mEnergyT[curMol]=0.0;
//        gBox.mVirialT[curMol]=0.0;
        gBox.mEnergy[curMol]=0.0;
        gBox.mVirial[curMol]=0.0;
//        sumE = 0.0;
//        sumV = 0.0;
        //intra_potential(curMol, gConf, gTop, gBox); //intramolecular energy
        for(int curMol2 = curMol; curMol2 < maxMol; curMol2++){  //get 
            //curMol2 = j; //gBox.fMol[blockIdx.x]+ i*blockDim.x + threadIdx.x;    //current molecule
            if(curMol != curMol2){
                inter_potential(curMol, curMol2, gConf, gTop, gBox, sumE, sumV);
            }
            else{
                intra_potential(curMol, gConf, gTop, gBox);
            }
        }
        gBox.mEnergyT[curMol] = gBox.mEnergy[curMol];
        gBox.mVirialT[curMol] = gBox.mVirial[curMol];
        //single_one_energy(yDim, curMol, gConf, gTop, gBox);
        //printf("nmol %d en %f vir %f \n", curMol, gBox.mEnergyT[curMol], gBox.mVirialT[curMol]);
    }
    __syncthreads();
  //sum potential   //change reduse
    if(threadIdx.x == 0){
        gBox.energy[blockIdx.x]=0.0;
        gBox.virial[blockIdx.x]=0.0;
        for(int i = gBox.fMol[blockIdx.x]; i < maxMol; i++){
            gBox.energy[blockIdx.x]+=gBox.mEnergyT[i];
            gBox.virial[blockIdx.x]+=gBox.mVirialT[i];
        }
        printf("0 mol %d last %d variant 1 total enegry %f virial %f \n", gBox.fMol[blockIdx.x], maxMol, gBox.energy[blockIdx.x], gBox.virial[blockIdx.x]);
    }
    __syncthreads();
    
// variant 2
    reduce = blockDim.x / 2;
//    gBox.energy[blockIdx.x]=0.0;
//    gBox.virial[blockIdx.x]=0.0;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            for(int i=0; i < yDim; i++){
                curMol2=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x;
                curMol=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x + reduce;
                gBox.mEnergyT[curMol2]+=gBox.mEnergyT[curMol];
                gBox.mVirialT[curMol2]+=gBox.mVirialT[curMol];
            }
        //printf(" reduce %d \n", reduce);
            reduce = reduce / 2;
        }
        __syncthreads();
        //
    }
    if(threadIdx.x == 0){
        gBox.energy[blockIdx.x] = 0.0;
        gBox.virial[blockIdx.x] = 0.0;
        for(int i=0; i<yDim; i++){
            gBox.energy[blockIdx.x] += gBox.mEnergyT[gBox.fMol[blockIdx.x]+blockDim.x*i];
            gBox.virial[blockIdx.x] += gBox.mVirialT[gBox.fMol[blockIdx.x]+blockDim.x*i];
        }
        //gBox.energy[blockIdx.x] = gBox.energy[blockIdx.x] / 2;
        //gBox.virial[blockIdx.x] = gBox.virial[blockIdx.x] / 2;
        //__syncthreads();
        printf("variant 2 total enegry %f virial %f \n", gBox.energy[blockIdx.x], gBox.virial[blockIdx.x]);
    }
    __syncthreads();
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
    __shared__ float rcut;
    float rtest;
    
    //molecule number
    int curAtomNumA;
    int curAtomNumB;
    int id;
    
    //printf(" a %d type %d b %d typeb %d\n", a ,gBox.mType[a], b, gBox.mType[b]);
    
    curAtomNumA=gTop.aNum[gBox.mType[a]];   //number of atoms in molecule A
    curAtomNumB=gTop.aNum[gBox.mType[b]];   //numbers of atoms in molecule B
    sumE=0.0f;
    sumV=0.0f;
    rcut=0.5f*gBox.boxLen[blockIdx.x];
    //printf("rcut %f\n", rcut);
    
    //molecule periodic boundary condition
    dx=(gBox.xm[b] - gBox.xm[a]);
    dy=(gBox.ym[b] - gBox.ym[a]);
    dz=(gBox.zm[b] - gBox.zm[a]);
    if(dx > rcut){  //
        dx = -gBox.boxLen[blockIdx.x] + dx;
    }
    if(dy > rcut ){
        dy = -gBox.boxLen[blockIdx.x] + dy;
    }
    if(dz > rcut){
        dz = -gBox.boxLen[blockIdx.x] + dz;
    }
    if(dx < -rcut){ //
        dx = gBox.boxLen[blockIdx.x] + dx;
    }
    if(dy < -rcut){
        dy = gBox.boxLen[blockIdx.x] + dy;
    }
    if(dz < -rcut){
        dz = gBox.boxLen[blockIdx.x] + dz;
    }
    if(dx*dx+dy*dy+dz*dz < rcut*rcut){
        for(int i = 0; i < curAtomNumA; i++){
            for(int j = 0; j < curAtomNumB; j++){
                id = gBox.aType[gBox.fAtom[b] + j] * gConf.potNum[0] + gBox.aType[gBox.fAtom[a] + i];
                xa = gBox.xa[gBox.fAtom[b] + j] - gBox.xa[gBox.fAtom[a] + i] + dx;
                ya = gBox.ya[gBox.fAtom[b] + j] - gBox.ya[gBox.fAtom[a] + i] + dy;
                za = gBox.za[gBox.fAtom[b] + j] - gBox.za[gBox.fAtom[a] + i] + dz;
                ra = xa*xa + ya*ya + za*za;
                ra = gTop.sigma[id] * gTop.sigma[id] / ra;
                ra = ra * ra * ra;  //6 power
                //calculate potential
                sumE += gTop.epsi[id] * (ra* ra - ra);
                sumV += gTop.epsi[id] * (6.0f*ra - 12.0f*ra*ra);
            }
        }
    }
    //
    //printf("a %d b %d ra %f rm %f E %f V %f\n", a, b, rtest, sqrt(dx*dx+dy*dy+dz*dz), sumE, sumV);
    gBox.mEnergy[a]+=sumE;
    gBox.mVirial[a]+=sumV;
    En = 0.0;
    Vir = 0.0;
    return 0;
}

__device__ int intra_potential(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    gBox.mEnergy[a]+=0.0;
    gBox.mVirial[a]+=0.0;
    return 0;
}

__device__ int single_conf_change(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox, curandState &devStates){
    float dx, dy, dz;
    //get type of change
    
    //molecule move
    dx = curand_uniform(&devStates)-0.5;
    dy = curand_uniform(&devStates)-0.5;
    dz = curand_uniform(&devStates)-0.5;
    gBox.xm[a]+=dx*gBox.transMaxMove[blockIdx.x];
    gBox.ym[a]+=dy*gBox.transMaxMove[blockIdx.x];
    gBox.zm[a]+=dz*gBox.transMaxMove[blockIdx.x];
    //
    if(gBox.xm[a]<0.0){
        gBox.xm[a]+=gBox.boxLen[blockIdx.x];
    }
    if(gBox.ym[a]<0.0){
        gBox.ym[a]+=gBox.boxLen[blockIdx.x];
    }
    if(gBox.zm[a]<0.0){
        gBox.zm[a]+=gBox.boxLen[blockIdx.x];
    }
    //
    if(gBox.xm[a]>gBox.boxLen[blockIdx.x]){
        gBox.xm[a]-=gBox.boxLen[blockIdx.x];
    }
    if(gBox.ym[a]>gBox.boxLen[blockIdx.x]){
        gBox.ym[a]-=gBox.boxLen[blockIdx.x];
    }
    if(gBox.zm[a]>gBox.boxLen[blockIdx.x]){
        gBox.zm[a]-=gBox.boxLen[blockIdx.x];
    }
    
    return 0;
}

__device__ int single_one_energy(int yDim, int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    __shared__ int maxMol;
    int curMol;
    int curMol2;
    float sumE;
    float sumV;
    __shared__ int reduce;

    maxMol = gBox.fMol[blockIdx.x] + blockDim.x*yDim;
    for(int i = 0; i < yDim; i++){    //for several molecules per thread
        curMol = gBox.fMol[blockIdx.x]+ i*blockDim.x + threadIdx.x;    //current molecule
        gBox.mEnergy[curMol]=0.0;
        gBox.mVirial[curMol]=0.0;
        if(a!=curMol){  //intermolecular energy
            inter_potential(curMol, a, gConf, gTop, gBox, sumE, sumV);
        }
        else{
            intra_potential(curMol, gConf, gTop, gBox);
        }
    }
    //reduse array
    __syncthreads();
    reduce = blockDim.x / 2;
    while(reduce > 0){
        if(threadIdx.x < reduce){
            for(int i = 0; i < yDim; i++){
                curMol2=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x;
                curMol=gBox.fMol[blockIdx.x]+threadIdx.x+i*blockDim.x + reduce;
                gBox.mEnergy[curMol2]+=gBox.mEnergy[curMol];
                gBox.mVirial[curMol2]+=gBox.mVirial[curMol];
            }
        //printf(" reduce %d energy %f \n", reduce, gBox.mEnergy[reduce]);
        }
        reduce = reduce / 2;
        __syncthreads();
        //
    }
    if(threadIdx.x==0){
        for(int i=1; i < yDim; i++){
            gBox.mEnergy[gBox.fMol[blockIdx.x]]+=gBox.mEnergy[gBox.fMol[blockIdx.x]+i*blockDim.x];
            gBox.mVirial[gBox.fMol[blockIdx.x]]+=gBox.mVirial[gBox.fMol[blockIdx.x]+i*blockDim.x];
        }
        gBox.mEnergyT[a]=gBox.mEnergy[gBox.fMol[blockIdx.x]];
        gBox.mVirialT[a]=gBox.mVirial[gBox.fMol[blockIdx.x]];
    }
    //printf("nmol %d en %f vir %f\n", a, gBox.mEnergyT[a], gBox.mVirialT[a]);
    __syncthreads();
    return 0;
}

__device__ int single_get_prop(int yDim, gOptions gConf, gMolecula gTop, gSingleBox &gBox, int curId){
    //get energy
    //single_calc_totenergy(yDim, gConf, gTop, gBox);
    if(threadIdx.x==0){
        gBox.eqEnergy[blockIdx.x * EQBLOCKSIZE + curId] = gBox.energy[blockIdx.x] / gBox.molNum[blockIdx.x];
    
    // pressure p = nkT + 1/3 W Проверить коэффициенты
        gBox.eqPressure[blockIdx.x * EQBLOCKSIZE + curId] = gConf.Temp[blockIdx.x] * gBox.molNum[blockIdx.x] / gBox.boxVol[blockIdx.x] + gBox.virial[blockIdx.x] /3.0f / gBox.molNum[blockIdx.x];
//        printf("id %d energy %f pressure %f \n", curId, gBox.eqEnergy[blockIdx.x * EQBLOCKSIZE + curId], gBox.eqPressure[blockIdx.x * EQBLOCKSIZE + curId]);
    }
    //get virial
    
    //get density
    
    //get volume
    
    //get capacity
    
    
    
    return 0;
}

__device__ int single_change_trans(gOptions gConf, gMolecula gTop, gSingleBox &gBox){
    
    if(threadIdx.x==0){
        if((gBox.accept[blockIdx.x]+1)/(gBox.reject[blockIdx.x]+1)>0.6){
            gBox.transMaxMove[blockIdx.x]*=1.2;
        }
        if(gBox.transMaxMove[blockIdx.x]>gBox.boxLen[blockIdx.x]/2.0){
            gBox.transMaxMove[blockIdx.x]=gBox.boxLen[blockIdx.x]/2.0;
        }
    }
    if(threadIdx.x==1){
        if((gBox.accept[blockIdx.x]+1)/(gBox.reject[blockIdx.x]+1)<0.4){
            gBox.transMaxMove[blockIdx.x]*=0.8;
        }
        if(gBox.transMaxMove[blockIdx.x]<0.01){
            gBox.transMaxMove[blockIdx.x]=0.01;
        }
    }
    
    if(threadIdx.x==2){
        gBox.accept[blockIdx.x]=0;
        gBox.reject[blockIdx.x]=0;
    }
    __syncthreads;
    return 0;
}

