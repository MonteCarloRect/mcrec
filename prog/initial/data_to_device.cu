#include "../mcrec.h"

int data_to_device(gSingleBox &gBox, singleBox* &inputData, gOptions &gConf, options &config, gMolecula &gTop, potentialParam* Param, molecules* initMol){
    //allocate and copy data to GPU
    gSingleBox hostData;
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
    sum=0;  //calculate total number oof molecules
    for(int i=0;i<config.flowNum;i++){
        sum+=inputData[i].molNum;
    }
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
    cudaMalloc(&gBox.boxLen, config.flowNum* sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostData.boxLen[i]=inputData[i].boxLen;
    }
    cudaMemcpy(gBox.boxLen, hostData.boxLen, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    
    //energy malloc
    cuErr = cudaMalloc(&gBox.virial, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.virial memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr =cudaMalloc(&gBox.energy, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.energy memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.virial=(float*) malloc(config.flowNum*sizeof(float));
    hostData.energy=(float*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i < config.flowNum; i++){
        hostData.virial[i]=0.0;
        hostData.energy[i]=0.0;
    }
    hostData.mVirial=(float*) malloc(sum*sizeof(float));
    hostData.mEnergy=(float*) malloc(sum*sizeof(float));
    for(int i=0; i < sum; i++){
        hostData.mVirial[i]=0.0;
        hostData.mEnergy[i]=0.0;
    }
    cuErr = cudaMemcpy(gBox.virial, hostData.virial, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
        }
    cuErr = cudaMemcpy(gBox.energy, hostData.energy, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
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
    cuErr = cudaMemcpy(gBox.mEnergy, hostData.mEnergy, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gBox.mVirial, hostData.mVirial, sum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    //transformation
    cuErr =cudaMalloc(&gBox.curMol, config.flowNum*sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.curMol memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.curMol=(int*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostData.curMol[i]=0;
    }
    cuErr = cudaMemcpy(gBox.curMol, hostData.curMol, config.flowNum*sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    cuErr =cudaMalloc(&gBox.transMaxMove, config.flowNum*sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate box.transMaxMove memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    hostData.transMaxMove=(float*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostData.transMaxMove[i]=0.2;
    }
    cuErr = cudaMemcpy(gBox.transMaxMove, hostData.transMaxMove, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    //============================TOPOLOGY
    gMolecula hostTop;
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
    for(int i=0; i<config.potNum; i++){
        for(int j=0; j<config.potNum; j++){
            id=i*config.potNum+j;
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
    cuErr = cudaMalloc(&gTop.sigma, config.potNum * config.potNum * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate top.sigma memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gTop.epsi, config.potNum * config.potNum * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate top.epsilon memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gTop.charge, config.potNum * config.potNum * sizeof(float));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate top.charge memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMalloc(&gTop.aNum, config.subNum * sizeof(int));
    if(cuErr != cudaSuccess){
        printf("Cannot allocate top.aNum memory file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gTop.sigma, hostTop.sigma, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gTop.epsi, hostTop.epsi, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gTop.charge, hostTop.charge, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    cuErr = cudaMemcpy(gTop.aNum, hostTop.aNum, config.subNum * sizeof(int), cudaMemcpyHostToDevice);
    if(cuErr != cudaSuccess){
        printf("Cannot copy memory to device file %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    
    
    //============================CONFIG
    gOptions hostConf;
    hostConf.Temp=(float*) malloc(config.flowNum*sizeof(float));
    hostConf.potNum=(int*) malloc(sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        hostConf.Temp[i]=config.flowT[i];
    }
    hostConf.potNum[0]=config.potNum;   //to arrays
    cudaMalloc(&gConf.Temp, config.flowNum*sizeof(float));
    cudaMemcpy(gConf.Temp, hostConf.Temp, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gConf.potNum, sizeof(int));
    cudaMemcpy(gConf.potNum, hostConf.potNum, sizeof(int), cudaMemcpyHostToDevice);
    
    
    return 0;
}



