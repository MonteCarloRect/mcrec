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

//GPU data
gSingleBox gBox;
gOptions gConf;
gMolecula gTop;


