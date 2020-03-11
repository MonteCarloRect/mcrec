#include "../mcrec.h"

int data_to_device(gSingleBox &gBox, singleBox* &inputData, gOptions* &gConf, options &config, gMolecula* &gTop, potentialParam* Param, molecules* initMol, gSingleBox &hostData, gMolecula &hostTop, gOptions &hostConf, int deviceCount, gDoublebox* &gDBox){
    //allocate and copy data to GPU
    //gSingleBox hostData;
    int sum;
    int id;
    cudaError_t cuErr;
    
    //===========================MOLECULES
    //numbers of molecules
    hostData.molNum=(int*)malloc(config.flowNum * sizeof(int));
    for(int i=0;i<config.flowNum;i++){
        hostData.molNum[i]=inputData[i].molNum;
    }
    cuErr = cudaMalloc(&gBox.molNum, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.molNum memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.molNum, hostData.molNum, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    //coordinats of molecules
    sum=0;  //calculate total number of molecules
    for(int i=0; i<config.flowNum; i++){
        sum+=inputData[i].molNum;
    }
    hostData.tMol = sum;
    hostData.xm=(float*)malloc(sum * sizeof(float));
    hostData.ym=(float*)malloc(sum * sizeof(float));
    hostData.zm=(float*)malloc(sum * sizeof(float));
    hostData.mType=(int*)malloc(sum * sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<inputData[i].molNum; j++){
            id=i*config.flowNum+j;
            hostData.xm[id]=inputData[i].xm[j];
            hostData.ym[id]=inputData[i].ym[j];
            hostData.zm[id]=inputData[i].zm[j];
            hostData.mType[id]=inputData[i].type[j];
            //printf("id %d xm %f ym %f zm %f type %d \n", id, hostData.xm[id], hostData.ym[id], hostData.zm[id], hostData.mType[id] );
        }
    }
    cuErr = cudaMalloc(&gBox.xm,  sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.xm memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.ym,  sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.ym memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.zm,  sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.zm memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.mType,  sum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.mType memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    cuErr = cudaMemcpy(gBox.xm, hostData.xm, sum*sizeof(float), cudaMemcpyHostToDevice);
    cuErr = cudaMemcpy(gBox.ym, hostData.ym, sum*sizeof(float), cudaMemcpyHostToDevice);
    cuErr = cudaMemcpy(gBox.zm, hostData.zm, sum*sizeof(float), cudaMemcpyHostToDevice);
    cuErr = cudaMemcpy(gBox.mType, hostData.mType, sum*sizeof(int), cudaMemcpyHostToDevice);
    //numbers of molecules of each type
    sum=0;
    for(int i=0; i<config.flowNum; i++){
        sum+=config.subNum;
    }
    hostData.typeMolNum=(int*) malloc (sum * sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<config.subNum; j++){
            id=config.flowNum*i+j;
            hostData.typeMolNum[id]=inputData[i].typeMolNum[j];
        }
    }
    cudaMalloc(&gBox.typeMolNum,  sum*sizeof(int));
    cudaMemcpy(gBox.typeMolNum, hostData.typeMolNum, sum*sizeof(int), cudaMemcpyHostToDevice);
    
    //type of atoms
    
    
    
    //atoms
    sum=0;
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<config.subNum; j++){
            sum+=inputData[i].typeMolNum[j];
        }
    }
    hostData.tAtom = sum;
    printf(" 1 total numbers of atoms %d\n", sum);
    hostData.xa=(float*) malloc(sum*sizeof(float));
    hostData.ya=(float*) malloc(sum*sizeof(float));
    hostData.za=(float*) malloc(sum*sizeof(float));
    hostData.aType=(int*) malloc(sum*sizeof(int));
    id=0;
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<inputData[i].molNum; j++){
            for(int k=0; k<inputData[i].aNum[j]; k++){
                //id=i*(config.flowNum*config.subNum)+j*config.subNum+k;
                hostData.xa[id]=inputData[i].xa[j][k];
                hostData.ya[id]=inputData[i].ya[j][k];
                hostData.za[id]=inputData[i].za[j][k];
                hostData.aType[id]=inputData[i].aType[j][k];
                //printf("test 223 %d \n", hostData.aType[id]);
                id++;
            }
        }
    }
    cuErr = cudaMalloc(&gBox.xa, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.xa memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.ya, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.ya memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.za, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.za memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.aType, sum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.aType memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.xa, hostData.xa, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.ya, hostData.ya, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.za, hostData.za, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.aType, hostData.aType, sum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    
    //get first molecules and atoms of flows
    hostData.fMol=(int*) malloc(config.flowNum*sizeof(int));
    cudaMalloc(&gBox.fMol, config.flowNum*sizeof(int));
    hostData.fMol[0]=0; //first molecule index 0
    sum=inputData[0].molNum;
    for(int i=1; i<config.flowNum; i++){
        //printf("my test %d", i);
        hostData.fMol[i]=hostData.fMol[i-1]+inputData[i].molNum;
        sum+=inputData[i].molNum;
    }
    printf(" 2 total numbers of atoms %d\n", sum);
    cudaMemcpy(gBox.fMol, hostData.fMol, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    hostData.fAtom=(int*) malloc(sum*sizeof(int));
    cuErr = cudaMalloc(&gBox.fAtom, sum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.fAtom memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.fAtom[0]=0;    //first atom
    id=0;
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<inputData[i].molNum; j++){
            //printf("my test %d %d\n",i,j);
            //molecule type --- inputData[i].type[j]
            if(id!=0){
                hostData.fAtom[id]=hostData.fAtom[id-1]+initMol[inputData[i].type[j]].atomNum;
                //printf("test data %d\n", hostData.fAtom[id]);
            }
            id++;
        }
    }
    cuErr = cudaMemcpy(gBox.fAtom, hostData.fAtom, sum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //box length
    hostData.boxLen=(float*) malloc(config.flowNum*sizeof(float));
    hostData.boxVol=(float*) malloc(config.flowNum*sizeof(float));
    cudaMalloc(&gBox.boxLen, config.flowNum * sizeof(float));
    cudaMalloc(&gBox.boxVol, config.flowNum * sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostData.boxLen[i]=inputData[i].boxLen;
        hostData.boxVol[i]=inputData[i].boxLen*inputData[i].boxLen*inputData[i].boxLen;
    }
    cudaMemcpy(gBox.boxLen, hostData.boxLen, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.boxVol, hostData.boxVol, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    //energy malloc
    cuErr = cudaMalloc(&gBox.virial, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.virial memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.energy, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.energy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.pressure, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.pressure memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    cuErr =cudaMalloc(&gBox.oldEnergy, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.oldEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.oldVirial, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.oldVirial memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.newEnergy, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.newEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.newVirial, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.newVirial memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.virial=(float*) malloc(config.flowNum*sizeof(float));
    hostData.energy=(float*) malloc(config.flowNum*sizeof(float));
    hostData.pressure=(float*) malloc(config.flowNum*sizeof(float));
    hostData.oldEnergy=(float*) malloc(config.flowNum*sizeof(float));
    hostData.oldVirial=(float*) malloc(config.flowNum*sizeof(float));
    hostData.newEnergy=(float*) malloc(config.flowNum*sizeof(float));
    hostData.newVirial=(float*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i < config.flowNum; i++){
        hostData.virial[i]=0.0;
        hostData.energy[i]=0.0;
        hostData.pressure[i] = 0.0f;
        hostData.oldEnergy[i]=0.0;
        hostData.oldVirial[i]=0.0;
        hostData.newEnergy[i]=0.0;
        hostData.newVirial[i]=0.0;
    }
    hostData.mVirial=(float*) malloc(sum*sizeof(float));
    hostData.mEnergy=(float*) malloc(sum*sizeof(float));
    hostData.mVirialT=(float*) malloc(sum*sizeof(float));
    hostData.mEnergyT=(float*) malloc(sum*sizeof(float));
    for(int i=0; i < sum; i++){
        hostData.mVirial[i]=0.0;
        hostData.mEnergy[i]=0.0;
        hostData.mVirialT[i]=0.0;
        hostData.mEnergyT[i]=0.0;
    }
    cuErr = cudaMemcpy(gBox.virial, hostData.virial, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.energy, hostData.energy, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.pressure, hostData.pressure, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.oldEnergy, hostData.oldEnergy, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.oldVirial, hostData.oldVirial, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.newEnergy, hostData.newEnergy, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.newVirial, hostData.newVirial, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        
    cuErr = cudaMalloc(&gBox.mEnergy, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.mEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.mVirial, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.mVirial memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.mEnergyT, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.mEnergyT memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.mVirialT, sum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.mVirialT memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.mEnergy, hostData.mEnergy, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.mVirial, hostData.mVirial, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.mEnergyT, hostData.mEnergyT, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.mVirialT, hostData.mVirialT, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //transformation
    cuErr =cudaMalloc(&gBox.curMol, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.curMol memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.accept, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.accept memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.reject, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.reject memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.tAccept, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.tAccept memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.tReject, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.tReject memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.curMol=(int*) malloc(config.flowNum*sizeof(int));
    hostData.accept = (int*) malloc(config.flowNum*sizeof(int));
    hostData.reject = (int*) malloc(config.flowNum*sizeof(int));
    hostData.tAccept = (int*) malloc(config.flowNum * sizeof(int));
    hostData.tReject = (int*) malloc(config.flowNum * sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        hostData.curMol[i]=0;
        hostData.accept[i]=0;
        hostData.reject[i]=0;
        hostData.tReject[i]=0;
        hostData.tAccept[i]=0;
    }
    cuErr = cudaMemcpy(gBox.curMol, hostData.curMol, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.accept, hostData.accept, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.reject, hostData.reject, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.tAccept, hostData.tAccept, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.tReject, hostData.tReject, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    cuErr =cudaMalloc(&gBox.transMaxMove, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.transMaxMove memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.transMaxMove=(float*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostData.transMaxMove[i]=0.05;
    }
    cuErr = cudaMemcpy(gBox.transMaxMove, hostData.transMaxMove, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //------------------block enegry
    cuErr =cudaMalloc(&gBox.eqBlockEnergy, config.flowNum * EQBLOCKS * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.eqBlockEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.eqBlockPressure, config.flowNum * EQBLOCKS * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.eqBlockEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.avEnergy, config.flowNum * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.avEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gBox.avPressure, config.flowNum * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.avPressure memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.eqBlockEnergy = (float*) malloc(config.flowNum * EQBLOCKS * sizeof(float));
    hostData.eqBlockPressure = (float*) malloc(config.flowNum * EQBLOCKS * sizeof(float));
    hostData.avEnergy = (float*) malloc(config.flowNum * sizeof(float));
    hostData.avPressure = (float*) malloc(config.flowNum * sizeof(float));
    hostData.energyCorr = (float*) malloc(config.flowNum * EQBLOCKS * sizeof(float));
    hostData.pressureCorr = (float*) malloc(config.flowNum * EQBLOCKS * sizeof(float));
    for(int i = 0; i < config.flowNum * EQBLOCKS; i++){
        hostData.eqBlockEnergy[i]=0.0;
        hostData.eqBlockPressure[i]=0.0;
    }
    for(int i = 0; i < config.flowNum; i++){
        hostData.avEnergy[i] = 0.0f;
        hostData.avPressure[i] = 0.0f;
        //printf("testtststs  %f   %f", hostData.avEnergy[i], hostData.avPressure[i]);
    }
    cuErr = cudaMemcpy(gBox.eqBlockEnergy, hostData.eqBlockEnergy, config.flowNum * EQBLOCKS * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.eqBlockPressure, hostData.eqBlockPressure, config.flowNum * EQBLOCKS * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.avEnergy, hostData.avEnergy, config.flowNum * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        //return 2;
    }
    cuErr = cudaMemcpy(gBox.avPressure, hostData.avPressure, config.flowNum * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //---------------------block size
    cuErr =cudaMalloc(&gBox.eqEnergy, config.flowNum * EQBLOCKSIZE * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.eqEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.eqPressure, config.flowNum * EQBLOCKSIZE * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.eqEnergy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.eqEnergy = (float*)malloc(config.flowNum * EQBLOCKSIZE * sizeof(float));
    hostData.eqPressure = (float*)malloc(config.flowNum * EQBLOCKSIZE * sizeof(float));
    for(int i=0; i < config.flowNum * EQBLOCKSIZE; i++){
        hostData.eqEnergy[i] = 0.0;
        hostData.eqPressure[i] = 0.0;
    }
    cuErr = cudaMemcpy(gBox.eqEnergy, hostData.eqEnergy, config.flowNum * EQBLOCKSIZE * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.eqPressure, hostData.eqPressure, config.flowNum * EQBLOCKSIZE * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    //============================TOPOLOGY
    //matrix
    hostTop.sigma=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    hostTop.epsi=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    hostTop.charge=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    //single atom
    hostTop.aNum=(int*) malloc (config.subNum * sizeof(int));
    for(int i=0; i<config.subNum; i++){
        
        hostTop.aNum[i]=initMol[i].atomNum;
        printf("test topology %d %d \n", i, hostTop.aNum[i]);
    }
    
//    printf("potnum %d\n", config.potNum);
    for(int i = 0; i < config.potNum; i++){
        for(int j = 0; j < config.potNum; j++){
            id = i * config.potNum + j;
//            printf("id %d i %d j %f\n", id, i, Param[i].sigma);
            //add mixture rule
            hostTop.sigma[id]=0.5*(Param[i].sigma+Param[j].sigma);
//            printf("sigma %f\n", hostTop.sigma[id]);
            hostTop.epsi[id]=4.0f*sqrt(Param[i].epsilon*Param[j].epsilon)*1000.0/R; //from kJ/mol to [K] kB
//            printf("epsi %f\n", hostTop.epsi[id]);
            hostTop.charge[id]=Param[i].charge*Param[j].charge;
//            printf("charge %f\n", hostTop.charge[id]);
            //add mixture rule
            printf("sigma %f  %f   %f\n", hostTop.sigma[id] ,Param[i].sigma, Param[j].sigma);
            printf("epsilon %f  %f   %f\n", hostTop.epsi[id] ,Param[i].epsilon, Param[j].epsilon);
            printf("charge %f  %f    %f\n", hostTop.charge[id] ,Param[i].charge, Param[j].charge);
        }
    }
//    printf("OLOLO5\n");
    //gpu data
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaStreamCreate(&gDBox[curDev].stream);  //create stream
        if(cuErr != cudaSuccess){
            printf("Cannot create stream on device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        
        cuErr = cudaMalloc(&gTop[curDev].sigma, config.potNum * config.potNum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate top[curDev].sigma memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gTop[curDev].epsi, config.potNum * config.potNum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate top[curDev].epsilon memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gTop[curDev].charge, config.potNum * config.potNum * sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate top[curDev].charge memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gTop[curDev].aNum, config.subNum * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate top[curDev].aNum memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gTop[curDev].sigma, hostTop.sigma, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gTop[curDev].epsi, hostTop.epsi, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gTop[curDev].charge, hostTop.charge, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gTop[curDev].aNum, hostTop.aNum, config.subNum * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        //set atom types
        hostTop.aType = (int*) malloc(config.subNum * MAXATOM * sizeof(int));
        cuErr = cudaMalloc(&gTop[curDev].aType, config.subNum * MAXATOM * sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate top[curDev].aType memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        for(int i = 0; i < config.subNum; i++){
            for(int j = 0; j < MAXATOM; j++){
                if(j < initMol[i].atomNum){
                    hostTop.aType[i * MAXATOM + j] = initMol[i].aType[j];
                }
                else{
                    hostTop.aType[i * MAXATOM + j] = 0.0;
                }
                
            }
        }
        cuErr = cudaMemcpy(gTop[curDev].aType, hostTop.aType, config.subNum  * MAXATOM * sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    }
    //============================CONFIG
    hostConf.Temp=(float*) malloc(config.flowNum*sizeof(float));
    hostConf.potNum=(int*) malloc(sizeof(int));
    hostConf.subNum=(int*) malloc(sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        hostConf.Temp[i]=config.flowT[i];
    }
    hostConf.potNum[0]=config.potNum;   //to arrays
    hostConf.subNum[0] = config.subNum; //to array
    printf("hostconf %d \n", hostConf.potNum[0]);
    for(int curDev = 0; curDev < deviceCount; curDev++){
        cuErr = cudaSetDevice(curDev);  //set to current device
        if(cuErr != cudaSuccess){
            printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gConf[curDev].subNum, sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gConf[curDev].subNum memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gConf[curDev].Temp, config.flowNum*sizeof(float));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gConf[curDev].Temp memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gConf[curDev].subNum, hostConf.subNum, sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gConf[curDev].Temp, hostConf.Temp, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMalloc(&gConf[curDev].potNum, sizeof(int));
        if(cuErr != cudaSuccess){
            printf("Cannot allocate gConf[curDev].potNum memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
        cuErr = cudaMemcpy(gConf[curDev].potNum, hostConf.potNum, sizeof(int), cudaMemcpyHostToDevice);
        if(cuErr != cudaSuccess){
            printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    }
    
    return 0;
}



