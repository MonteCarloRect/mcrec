//varaibles
//#include "mcrec.h"

options config;
int deviceCount;
cudaDeviceProp* deviceProp;
molecules* initMol; //inserted molecules
FILE* logFile;
singleBox* gpuSingleBox;
singleBox* initFlows;
potentialParam* allParams;
int paramsLines;
potentialParam* gpuParams;
potentialParam* hostParams;
mixParam** gpuMixParams;
mixParam** hostMixParams;

gSingleBox hostData;
gMolecula hostTop;

//GPU data
cudaError_t cuErr;
gSingleBox gBox;
gOptions gConf;
gMolecula gTop;

//time
time_t beginTime;
time_t endTime;
